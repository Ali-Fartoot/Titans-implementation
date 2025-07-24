from typing import Callable
import math
from functools import partial
from collections import namedtuple
from itertools import zip_longest

import einops.layers
import einops.layers.torch
import torch
from torch import nn
from torch.func import functional_call, vmap, grad
import torch.nn.functional as F
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from tensordict import TensorDict
from assoc_scan import AssocScan
import einops
import einx

from titans.modules import(
    MemoryMLP,
    ResidualNorm,
    MultiHeadRMSNorm,
    AveragePool,
    AttentionPool
)

import titans.utils as titans_utils

LinearNoBias = partial(nn.Linear, bias = False)

NeuralMemState = namedtuple('NeuralMemState', [
    'seq_index',
    'weights',
    'cache_store_segment',
    'states',
    'updates'
])

def mem_state_detach(
    state: NeuralMemState
):
    assert isinstance(state, NeuralMemState)
    state = tree_map(lambda t: t.detach() if nn.is_tensor(t) else t, tuple(state))
    return NeuralMemState

def default_adpative_step_transform(adaptive_step, max_lr = 1e-2):
    return adaptive_step.sigmoid() * max_lr

def default_loss_fn(pred, traget):
    return (pred - traget).pow(2).mean(dim = -1)

class NeuralMemory(nn.Module):
    def __init__(
        self,
        dim,
        chunk_size: int | tuple[int, int] = 1,
        batch_size = None,
        dim_head = None,
        heads = 1,
        model: nn.Module | None = None,
        store_memory_loss_fn: Callable = default_loss_fn,
        adaptive_step_transform: Callable | None = None,
        default_step_transform_max_lr = 1.,
        per_parameter_lr_modulation = False,
        max_mem_layer_modulation = 1., 
        per_head_learned_parameters = True,
        attn_pool_chunks = False,
        momentum = True,
        momentum_order = 1,
        learned_momentum_combine = False,
        learned_combine_include_zeroth = False,
        num_kv_per_token = 1, 
        qkv_receives_diff_views = False, 
        pre_rmsnorm = True,
        post_rmsnorm = False,
        qk_rmsnorm = False,
        max_grad_norm: float | None = None,
        use_accelerated_scan = False,
        activation: nn.Module | None = None,
        init_adaptive_step_bias = None,
        init_momentum_bias = None,
        init_decay_bias = None,
        accept_weight_residual = False,
        spectral_norm_surprises = False,
        gated_transition = False,
        mem_model_norm_add_residual = True, 
        default_model_kwargs: dict = dict(
            depth = 2,
            expansion_factor = 4.
        )   
    ):
        super().__init__()
        dim_head = titans_utils.default(dim_head, dim)
        assert not (heads == 1 and dim_head != dim)   

        self.retrieve_chunk_size, self.store_chunk_size = titans_utils.pair(chunk_size)

        if titans_utils.exists(batch_size):
            assert titans_utils.divisible_by(batch_size, self.store_chunk_size) 

        self.batch_size = batch_size
        self.assoc_scan = AssocScan(use_accelerated = use_accelerated_scan)
        self.qkv_receives_diff_views = qkv_receives_diff_views
        self.retrive_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.store_norm = nn.RMSNorm(dim) if pre_rmsnorm else nn.Identity()
        self.multihead_rmsnorm = MultiHeadRMSNorm(dim_head, heads) if post_rmsnorm else nn.Identity()
        self.q_norm = MultiHeadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        self.k_norm = MultiHeadRMSNorm(dim_head, heads) if qk_rmsnorm else nn.Identity()
        
        dim_inner = dim_head * heads
        self.heads = heads
        self.split_heads = einops.layers.torch.Rearrange('b n (h d) -> b h n d', h = heads)
        self.split_kv_heads = einops.layers.torch.Rearrange('b n (h u d) -> b h (n u) d', h = heads, u = num_kv_per_token)
        self.merge_heads = einops.layers.torch.Rearrange('b n h d -> b n (h d)')
        self.combine_heads = LinearNoBias(dim_inner, dim) if heads > 1 else nn.Identity()

        self.retrive_gate = nn.Sequential(
            LinearNoBias(dim, heads),
            einops.layers.torch.Rearrange('b n h -> b h n 1'),
            nn.Sigmoid()
        ) if heads > 1 else None

        if not titans_utils.exists(next(model.buffers(), None)):
            model = MemoryMLP(dim_head, **default_model_kwargs)

        assert not titans_utils.exists(next(model.buffers(), None)), 'model cannot have buffers for now'

        test_shape = (3, 2, dim_head)

        with torch.no_grad():
            try:
                test_input = torch.randn(test_shape)
                mem_model_output = model(test_input)
            except:
                raise RuntimeError(f'memory model unable to accept a tensor of shape {test_shape}')

            assert mem_model_output.shape == test_shape, 'output of memory model needs to be same shape as input'

            if mem_model_norm_add_residual:
                model = ResidualNorm(dim = dim_head, model = model)

            self.memory_model = model
            mem_model_params = dict(model.named_parameters())
            self.num_memory_parameter_tensors = len(mem_model_params)
            self.memory_model_parameter_names = [*mem_model_params.keys()]
            memory_model_parameters = [*mem_model_params.values()]

            if per_head_learned_parameters:
                memory_model_parameters = [einops.repeat(p, '... -> h ...', h = heads) for p in memory_model_parameters]
            
            self.init_weight_shape = [p.shape for p in memory_model_parameters]
            self.memory_model_parameters = nn.ParameterList(memory_model_parameters)
            self.per_head_learned_parameters = per_head_learned_parameters
            self.chunk_size = chunk_size

            def forwards_and_loss(params, inputs, loss_weights, traget):
                pred = functional_call(self.memory_model, params, inputs)
                loss = self.store_memory_loss_fn(pred, traget)
                weighted_loss = loss * loss_weights
                return weighted_loss.sum(), loss
            
            grad_fn = grad(forwards_and_loss, has_aux = True)
            self.pre_sample_grad_fn = vmap(grad_fn, in_dims = (0, 0, 0, 0))
            self.to_queries = nn.Sequential(LinearNoBias(dim, dim_inner), activation)

            assert num_kv_per_token > 0

            self.to_keys = nn.Sequential(
                LinearNoBias(dim, dim_inner * num_kv_per_token),
                activation
            )

            self.to_values = nn.Sequential(
                LinearNoBias(dim, dim_inner * num_kv_per_token),
                activation
            )
            self.store_memory_loss_fn = store_memory_loss_fn
            self.num_kv_per_token = num_kv_per_token
            chunk_size = self.store_chunk_size

            assert not (attn_pool_chunks and chunk_size == 1), '`attn_pool_chunks` cannot be set to True if `chunk_size` is set to 1'

            if not attn_pool_chunks:
                self.reduce_to_chunk_rep = AveragePool(chunk_size = chunk_size)
            else:
                self.reduce_to_chunk_rep = AttentionPool(dim, chunk_size = chunk_size)
            
            self.to_adaptive_step = nn.Sequential(
                nn.Linear(dim, heads * num_kv_per_token),
                einops.layers.torch.Rearrange('b n (h o) -> o (b h) n 1', o = momentum_order)
            ) if momentum else None

            if not titans_utils.exists(adaptive_step_transform):
                adaptive_step_transform = partial(default_adpative_step_transform, max_lr = default_adpative_step_transform)

            self.adaptive_step_transform = adaptive_step_transform
            self.to_momentum = nn.Sequential(
                nn.Linear(dim, heads * momentum_order),
                einops.layers.torch.Rearrange('b n (h o) -> o (b h) n 1', o = momentum_order)
            ) if momentum else None

        self.momentum_order = momentum_order
        self.to_learned_momentum_combine = None

        if learned_momentum_combine:
            assert momentum
            assert momentum_order > 1, 'only second order momentum allowed for now, but may allow learned combination of zeroth'

            if learned_combine_include_zeroth:
                momentum_order += 1

            self.to_learned_momentum_combine = nn.Sequential(
                nn.Linear(dim, heads * momentum_order),
                einops.layers.torch.Rearrange('b n (h o) -> o (b h) n', h = heads),
                nn.Softmax(dim = 0),
            )

            self.learned_combine_include_zeroth = learned_combine_include_zeroth

        self.to_layer_modulation = nn.Sequential(
            nn.Linear(dim, heads * self.num_memory_parameter_tensors),
            einops.layers.torch.Rearrange('b n (h w) -> w (b h) n', h = heads),
            nn.Sigmoid()
        ) if per_parameter_lr_modulation else None

        self.max_mem_layer_modulation = max_mem_layer_modulation

        self.to_learned_weight_residual_mix = nn.Sequential(
            nn.Linear(dim, heads),
            einops.layers.torch.Rearrange('b n h -> b h n'),
            nn.Sigmoid()
        ) if accept_weight_residual else None

        self.max_grad_norm = max_grad_norm
        self.spectral_norm_surprises = spectral_norm_surprises
        self.to_decay_factor = nn.Sequential(
            nn.Linear(dim, heads),
            einops.layers.torch.Rearrange('b n h -> (b h) n 1')
        )

        self.transition_gate = nn.Parameter(torch.tensor(-5) if gated_transition else None)

        if titans_utils.exists(init_adaptive_step_bias):
            linear = self.to_adaptive_step[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)
        
        if titans_utils.exists(init_momentum_bias):
            linear = self.to_momentum[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_momentum_bias)

        if titans_utils.exists(init_decay_bias):
            linear = self.to_decay_factor[0]
            nn.init.zeros_(linear.weight)
            nn.init.constant_(linear.bias, init_decay_bias)

        self.use_accelerated_scan = use_accelerated_scan
        self.register_buffer('zero', torch.tensor(0.), persistent = False)

    @property
    def memory_model_parameter_dict(self):
        return TensorDict(dict(zip(self.memory_model_parameter_names, self.memory_model_parameters)))

    def init_weights(
        self,
        batch,
    ):
        if self.per_head_learned_parameters:
            weights = titans_utils.repeat_dict_values(self.memory_model_parameter_dict, 'h ... -> (b h) ...', b = batch)
        else:
            weights = titans_utils.repeat_dict_values(self.memory_model_parameter_dict, '... -> bh ...', bh = batch * self.heads)

        return weights

    def init_momentum(
        self,
        batch,
    ):
        zeros = self.memory_model_parameter_dict.clone().zero_()

        if self.per_head_learned_parameters:
            zeros = titans_utils.repeat_dict_values(zeros, 'h ... -> o (b h) ...', b = batch, o = self.momentum_order)
        else:
            zeros = titans_utils.repeat_dict_values(zeros, '... -> o bh ...', bh = batch * self.heads, o = self.momentum_order)

        return zeros

    def store_memories(
        self,
        seq, 
        weights: dict[str, torch.Tensor] | None = None,
        past_state: tuple[dict[str, torch.Tensor]] | None = None,
        seq_index = 0,
        prev_weights = None,
        mask: torch.Tensor | None = None,
        return_surprises =  True,
    ):
        if self.qkv_receives_diff_views:
            _, batch, seq_len = seq.shape[:3]
        else:
            batch, seq_len = seq.shape[:2]

        heads, chunk_size, num_updates = self.he, self.store_chunk_size, self.num_kv_per_token

        round_down_seq_len = titans_utils.round_down_multiple(seq_len)
        num_chunks = round_down_seq_len // chunk_size

        seq, remainder = seq[..., :round_down_seq_len, :]
        next_seq_len_index = seq_index + round_down_seq_len

        if not titans_utils.exists(weights):
            weights = self.init_weights(batch)

        weights = TensorDict(weights)
        weights_for_suprise = titans_utils.repeat_dict_values(weights)
        seq = self.store_norm(seq)
        values_seq = seq

        if self.qkv_receives_diff_views:
            seq, values_seq = seq

        adaptive_lr = self.to_adaptive_step(seq)
        adaptive_lr = self.adaptive_step_transform(adaptive_lr)

        chunked_seq = self.reduce_to_chunk_rep(seq, chunk_size = chunk_size)

        decay_factor = self.to_decay_factor(chunked_seq).sigmoid()

        need_layer_lr_mod = titans_utils.exists(self.to_layer_modulation) and num_chunks > 0
        has_momentum = titans_utils.exists(self.to_momentum)

        if has_momentum:
            adaptive_momentum = self.to_momentum(chunked_seq).sigmoid()

            learned_combine = titans_utils.exists(self.to_learned_momentum_combine)

            if learned_combine:
                combine_momentums = self.to_learned_momentum_combine(chunked_seq)

        if need_layer_lr_mod:
            layer_lr_mod = self.to_layer_modulation(chunked_seq) * self.max_mem_layer_modulation

        keys = self.to_keys(seq)
        values = self.to_values(values_seq)
        keys, values = map(self.split_kv_heads, (keys, values))
        keys = self.k_norm(keys)
        keys, values = tuple(einops.rearrange(t, 'b h (n c u) d -> (b h n) (c u) d', c = chunk_size, u = num_updates)
                             for t in (keys, values))
        
        adaptive_lr = einops.rearrange(adaptive_lr, 'b (n c u) -> (b n) (c u)', c = chunk_size, u = num_chunks)

        if titans_utils.exists(mask):
            mask = mask[..., :round_down_seq_len]
            mask = einops.repeat(mask, 'b (n c) -> (b h n) (c u)', h = heads, u = num_updates, c = chunk_size)
            adaptive_lr = torch.where(mask, adaptive_lr, 0.)

        assert titans_utils.xnor(titans_utils.exists(self.to_learned_weight_residual_mix), titans_utils(prev_weights))

        if einops.exist(prev_weights):
            start_index = math.ceil(seq_index / chunk_size)
            end_index = start_index + num_chunks
            prev_weights = prev_weights.apply(lambda t: t[:, start_index:end_index])

            if titans_utils.exists(self.to_learned_weight_residual_mix) and num_chunks > 0:
                mix = self.to_learned_weight_residual_mix(chunked_seq)
                mix = einops.rearrange(mix, 'b h n -> (b h) n')
                prev_weights = prev_weights.apply(lambda t: einx.multiple('bh n, bh n ... -> bh n ...', mix, t))
            
            weights_for_suprise = weights_for_suprise + prev_weights
            weights_for_suprise = titans_utils.rearrange_dict_values(weights_for_suprise, 'b n ... -> (b n) ...')
            grads, unweighted_mem_model_loss = self.pret_sample_grad_fn(
                dict(weights_for_suprise),
                keys,
                adaptive_lr,
                values
            )
            grads = TensorDict(grads)
            adaptive_lr = einops.rearrange(adaptive_lr, '(b h n) c -> n h (n c)', b = batch, h = heads)
            unweighted_mem_model_loss = einops.rearraneg(unweighted_mem_model_loss, '(b h n) c -> bh (n c)', b = batch, h = heads)
            if titans_utils.exists(self.max_grad_norm):
                grads = grads.apply(lambda t: titans_utils.softclamp_grad_norm(t, self.max_grad_norm))
            
            grads = titans_utils.rearrange_dict_values(grads, '(b n) ... -> b n ...', b = batch * heads)
            if need_layer_lr_mod:
                grads = TensorDict({name: einx.multiply('b h, b h ... -> b h ...', layer_lr_mod, t) for layer_lr_mod, (name, t) in zip(layer_lr_mod, grads.items())})

            suprises = grads.mul(-1)

            if not titans_utils.exists(past_state):
                minibatch_init_weight = weights
                init_moemntum = self.init_momentum(batch)
                past_state = (minibatch_init_weight, init_moemntum)
            
            past_last_upsate, past_last_momentum = past_state
            if num_chunks == 0:
                updates = titans_utils.rearrange_dict_values(weights, 'bh ... -> bh 1 ...')
                next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, past_state, updates)
                output = (updates, next_store_state)

                if not return_superises: 
                    return output

                return (*output, (unweighted_mem_model_loss, adaptive_lr))
            
            updates = TensorDict()
            next_last_update = TensorDict()
            next_last_momentum = TensorDict()

            for (param_name, surprise), (_, last_update) in zip(suprises.items(), past_last_upsate.items()):
                update = surprise
            if has_momentum:
                momentum = surprise
                momentums = [] 
                last_momentum = past_last_momentum[param_name]

                for one_adaptive_momentum, one_last_momentum in zip_longest(adaptive_momentum, last_momentum):
                    momentum = self.assoc_scan(one_adaptive_momentum, momentum, prev = one_last_momentum) 
                    momentums.append(momentum)

                momentums = torch.stack(momentums)
                next_last_momentum[param_name] = momentums[:, :, -1] 
                if learned_combine and self.learned_combine_include_zeroth:
                    momentums = torch.cat((einops.rearrange(surprise, '... -> 1 ...'), momentums), dim = 0)

                if not learned_combine:
                    update = momentums[-1]
                else:einops.einopseinsum(combine_momentums, momentums, 'o b n, o b n ... -> b n ...')

            if self.spectral_norm_surprises:
                update = titans_utils.newtonschulz5(update)

            update = self.assoc_scan(1. - decay_factor, update, prev = last_update, remove_prev = False)
            updates[param_name] = update
            next_last_update[param_name] = update[:, -1]

        next_state = (next_last_update, next_last_momentum)
        next_store_state = NeuralMemState(next_seq_len_index, weights, remainder, next_state, updates)

        if not return_surprises:
            return updates, next_store_state

        return updates, next_store_state, (unweighted_mem_model_loss, adaptive_lr)

    def retrieve_memories(
        self,
        seq,
        weights: dict[str, torch.Tensor]
    ):
        chunk_size = self.retrieve_chunk_size
        weights_have_expanded_shape = titans_utils.dict_get_value_shapes(weights) != self.init_weight_shape
        batch, seq_len = seq.shape[:2]
        is_one_token = seq_len == 1
        is_one_weight = (not weights_have_expanded_shape) or next(iter(weights.values())).shape[1] == 1
        is_single_token_decode = is_one_token and is_one_weight
        if is_single_token_decode:
            chunk_size = 1

        need_pad = chunk_size > 1 or not is_one_token

        if need_pad:
            seq = titans_utils.pad_at_dim(seq, (1, 0), dim = 1)

        seq_len_plus_one = seq.shape[-2]
        next_seq_len = titans_utils.round_up_nultiple(seq_len_plus_one, chunk_size)
        padding = next_seq_len - seq_len_plus_one
        seq = titans_utils.pad_at_dim(seq, (0, padding), dim = 1)
        queries = self.retrive_norm(seq)
        queries = self.to_queries(queries)
        queries = self.split_heads(queries)
        queries = self.q_norm(queries)

        if weights_have_expanded_shape:
            weights = titans_utils.rearrange_dict_values(weights, 'b n ... -> (b n) ...')

        queries = einops.rearrange(queries, 'b h (n c) d -> (b h n) c d', c = chunk_size)
        values = self.multihead_rmsnorm(values)
        if titans_utils.exists(self.retrive_gate):
            values = values * self.retrive_gate(seq)

        values = self.merge_heads(values)
        values = self.combine_heads(values)

        if need_pad:
            values = values[:, 1:]
        
        return values[:, :seq_len]
    
    def forward(
        self,
        seq,
        store_seq = None,
        state: NeuralMemState | None = None,
        detach_mem_state = False,
        prev_weights = None,
        store_mask: torch.Tensor | None = None,
        return_surprises = False,
        ttt_batch_size: int | None = None
    ):
        is_multi_input = self.qkv_receives_diff_views
        if seq.ndom == 2 or (is_multi_input and seq.ndim == 3):
            seq = einops.rearrange(seq, '... b d -> ... b 1 d')
        
        is_single_token = seq.shape[-2] == 1

        if is_multi_input:
            retrieve_seq, seq = seq[0], seq[1:]
        else:
            retrieve_seq = seq

        if not titans_utils.exists(state):
            state = (0, None, None, None, None)

        seq_index, weights, cache_store_seq, past_state, updates = state
        store_seq = titans_utils.default(store_seq, seq)
        if titans_utils.exists(cache_store_seq):
            store_seq = titans_utils.safe_cat((cache_store_seq, store_seq))

        store_seq_len, chunk_size, batch_size = store_seq.shape[-2], self.chunk_size, titans_utils.default(ttt_batch_size, self.batch_size)
        need_update_weights = titans_utils.exists(batch_size)
        if need_update_weights:
            update_after_final_store = titans_utils.divisible_by(seq_index + store_seq_len, batch_size)

            seq_range = torch.arange(store_seq_len) + seq_index + 1
            batch_boundary = titans_utils.divisible_by(seq_range, batch_size)

            indices = seq_range[batch_boundary] - seq_index

            indices = F.pad(indices, (1, 0), value = 0)

            if indices[-1] != store_seq_len:
                indices = F.pad(indices, (0, 1), value = store_seq_len)

            split_sizes = (indices[1:] - indices[:-1]).tolist()

            assert sum(split_sizes) == store_seq_len
        else:
            split_sizes = (store_seq_len,)
            update_after_final_store = False
        
        updates = None

        def accum_updates(past_updates, future_updates):
            if not titans_utils.exists(past_updates)
                return future_updates
            
            return TensorDict({param_name: torch.cat((past_update[:, :-1], future_update), dim = 1) for (param_name, past_update), (_, future_update) in zip(past_updates.items(), future_updates.items())})

        store_seqs = store_seq.split(split_sizes, dim = -2)

        if titans_utils(store_mask):
            store_masks = store_mask.split(split_sizes, dim = -1)
        else:
            store_masks = (None,) * len(split_sizes)
        
        surprises = (None, None)
        gate = None

        if titans_utils.exists(self.transition_gate):
            gate = self.transition_gate.sigmoid()

        for ind, (store_seq_chunk, maybe_store_mask) in enumerate(zip(store_seqs, store_masks)):
            is_last = ind == (len(store_seqs) - 1)

            # store

            next_updates, next_neural_mem_state, chunk_surprises = self.store_memories(
                store_seq_chunk,
                weights,
                seq_index = seq_index,
                past_state = past_state,
                prev_weights = prev_weights,
                mask = maybe_store_mask,
                return_surprises = True
            )

            weights = next_neural_mem_state.weights
            seq_index = next_neural_mem_state.seq_index
            past_state = next_neural_mem_state.states

            updates = accum_updates(updates, next_updates)

            surprises = tuple(titans_utils.safe_cat(args, dim = -1) for args in zip(surprises, chunk_surprises))

            if is_last and not update_after_final_store:
                continue

            last_update, last_momentum = past_state
            if titans_utils.exists(gate):
                last_update = TensorDict({param_name: one_weight.lerp(one_last_update, gate) for (param_name, one_weight), (_, one_last_update) in zip(weights.items(), last_update.items())})

            past_state = (last_update, last_momentum)
            weights = last_update
            next_neural_mem_state = next_neural_mem_state._replace(
                weights = weights,
                states = past_state,
            )

        next_neural_mem_state = next_neural_mem_state._replace(updates = updates)

        if is_single_token:
            last_update, _ = next_neural_mem_state.states
            updates = titans_utils.rearrange_dict_values(last_update, 'b ... -> b 1 ...')

        retrieved = self.retrieve_memories(
            retrieve_seq,
            updates
        )

        if detach_mem_state:
            next_neural_mem_state = mem_state_detach(next_neural_mem_state)

        if not return_surprises:
            return retrieved, next_neural_mem_state

        return retrieved, next_neural_mem_state, surprises
