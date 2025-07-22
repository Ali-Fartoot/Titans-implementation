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
