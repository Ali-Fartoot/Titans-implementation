import torch
from torch import nn, cat
import torch.nn.functional as F
from utils import LayerNorm, l2norm
import titans.utils as titans_utils
import einops
from rotary_embedding_torch import RotaryEmbedding
from x_transformers.attend import Attend
from functools import partial
from einops.layers.torch import Rearrange
from collections import namedtuple
from typing import Callable
from torch.nn.attention.flex_attention import flex_attention


class ResidualNorm(nn.Module):
    def __init__(
            self,
            dim,
            model: nn.Module
    ):
        super().__init__()
        self.norm = LayerNorm(dim)
        self. model = model

    def forward(self, x):
        out = self.model(x)
        return self.norm(out) + x
    

class MemoryMLP(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            expansion_factor = 2.

    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim)
        
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, out_dim))
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])
        
        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
    
    def forward(
            self,
            x
    ):
        for ind, weight in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight

        return x
    
class GatedResidualMemoryMLP(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            expansion_factor = 4.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.randn(dim, dim_hidden)),
                nn.Parameter(torch.randn(dim_hidden, dim)),
                nn.Paramter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])

        for param in self.parameters():
            nn.init.xavier_uniform_(param)
        
    def forward(self, x):
        
        for weight1, weight2, to_gates in self.weights:
            res = x
            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2
            gates = cat((branch_out, res), dim=-1) @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        return x @ self.final_proj
    

class FactorizedMemoryMLP(nn.Module):
    def __init__(
            self, 
            dim,
            depth,
            k = 32,
    ):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.ParameterList([
                nn.Parameter(torch.randn(dim, k)),
                nn.Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)
    
    def forward(self, x):

        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight1 @ weight2

        return x
    

class MemorySwiGluMLP(nn.Module):
    def __init__(
        self,
        dim,
        depth = 1,
        expansion_factor = 4.
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        weights = []

        for _ in range(depth):
            weights.append(nn.ParameterList([
                nn.Parameter(torch.randn(dim, dim_inner * 2)),
                nn.Parameter(torch.randn(dim_inner, dim)),
            ]))

        self.weights = nn.ParameterList(weights)
        self.norm = LayerNorm(dim)

    def forward(self, x):

        for w1, w2 in self.weights:
            residual = x
            x, gates = (x @ w1).chunk(2, dim = -1)
            x = x * F.gelu(gates)
            x = x @ w2
            x = x + residual

        return self.norm(x)

class MemoryAttention(nn.Module):
    def __init__(
            self,
            dim,
            scale = 8.,
            expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(dim, dim)), # Q
            nn.Parameter(torch.randn(dim, dim)), # K
            nn.Parameter(torch.randn(dim, dim)), # V
            nn.Parameter(torch.randn(dim, dim_ff_hidden)),
            nn.Parameter(torch.randn(dim_ff_hidden, dim))
        ])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)
        
    def forward(self, x):
        wq, wk, wv, ffw1, ffw2 = self.weights

        q = l2norm(x @ ffw1)
        k = l2norm(x @ wk)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            is_causal = True
        )
        h = F.gelu(x @ ffw1)
        ff_out = h @ ffw2

        return attn_out + ff_out
    
class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.rmsnorm = nn.RMSNorm(dim, elementwise_affine = False)
        self.gamma = nn.Parameter(torch.zeros(heads, 1, dim))

    def forward(self, x):
        return self.rmsnorm(x) * self.gamma + 1.
    
class AveragePool(nn.Module):
    def __init__(
        self,
        chunk_size,             
    ):
        super().__init__()
        self.chunk_size = chunk_size
    
    def forward(
            self,
            x,
            chunk_size = None
    ):
            chunk_size = titans_utils.default(chunk_size, self.chunk_size)
            return einops.reduce(x, 'b (n c) d -> b n d', 'mean', c = chunk_size)
    
class AttentionPool(nn.Module):
    def __init__(
        self,
        dim,
        chunk_size
    ):
        chunk_size = titans_utils.default(chunk_size, self.chunk_size)
        x = einops.rearrange(x, 'b (n c) d -> b n c d', c = chunk_size)
        attn_logits = self.to_attn_logits(x)
        attn = attn_logits.softmax(dim = -2)
        return einops.reduce(x * attn, 'b n c d -> b n d', 'sum')
    

LinearNoBias = partial(nn.Linear, bias = False)

AttnIntermediates = nametuple('AttnIntermediates', ('value_residual', 'cached_key_values'))
class SegmentedAttention(nn.Module):
    def __init__(
            self,
            dim,
            segment_len,
            num_persist_mem_tokens = 0,
            num_longterm_mem_tokens = 0,
            dim_head = 64,
            heads = 8,
            sliding = False,
            accept_value_residual = False,
            attend_kwargs: dict = dict(),
            use_flex_atten = False
    ):
            super().__init__()
            self.norm = nn.RMSNorm(dim)
            dim_inner = dim_head * heads
            self.rotary_emb = RotaryEmbedding(dim_head)
            self.attend = Attend(causal=True, **attend_kwargs)
            self.to_qkv = LinearNoBias(dim, dim_inner * 3)
            self.to_out = LinearNoBias(dim_inner, dim)

            self.to_learned_v_mix = nn.Sequential(
                nn.Linear(dim, heads),
                Rearrange('b n h -> b h n 1'),
                nn.Sigmoid()
           ) if accept_value_residual else None
            
            self.segment_len = segment_len

            self.num_longterm_mem_tokens = num_longterm_mem_tokens
            total_segment_len = segment_len + num_longterm_mem_tokens
            self.total_segment_len = total_segment_len
            self.sliding = sliding
            self.split_heads = Rearrange('b n (h d) -> b h n d', h = heads)
            self.merge_heads = Rearrange('n h n d -> b n (h d)')
            self.persistent_memory = nn.Parameter(torch.zeros(2,
                                                              heads,
                                                              num_persist_mem_tokens,
                                                              dim_head
                                                              )
            )
            assert not (use_flex_atten and not titans_utils.exists(use_flex_atten))
            self.use_flex_attn = use_flex_atten
            self.segment_len = segment_len
            self.num_persist_mem_tokens = num_persist_mem_tokens

    def forward_infernce(
            self,
            token,
            cache,
            value_residual = None,
            output_gating = None
    ):
            batch = token.shape[0]
            token = self.norm(token)
            q, k, v = self.to_qkv(token).chunk(3, dim = -1)
            q, k, v = map(self.split_heads, (q, k, v))

            origin_v = v

            if titans_utils.exists(self.to_learned_v_mix):
                mix = self.to_learned_v_mix(token)
                v = v.lerp(value_residual. mix)
            
            next_cache = (k, v)
            q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
            q, k, v = tuple(einops.rearrange(t, 'b h n d -> n h n d') for t in (q, k, v))
            pmk, pmv = einops.repeat(self.persistent_memory,
                                     'kv ... -> kv b ...',
                                      b = k.shape[0]
            )
            k = torch.cat((pmk, k), dim = -2)
            v = torch.cat((pmv, v), dim = -2)
            out, _ = self.attend(q, k, v)
            out = self.merge_heads(out)
            out = self.to_out(out)

            if titans_utils.exists(output_gating):
                out = out * output_gating

            return out, AttnIntermediates(origin_v, next_cache)       
    def forward_flex(
            self,
            seq,
            value_residual = None,
            flex_attn_fn: Callable | None = None,
            output_gating = None,
            cache = None
    ):
        assert not (titans_utils.exists(value_residual) ^ 
                    titans_utils.exists(self.to_learned_v_mix))
        
        batch, seq_len = seq.shape[:2]
        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        orig_v = v

        if titans_utils.exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual. mix)
        
        next_cache = (k, v)
        pmk, pmv = einops.repeat(self.persistent_memory,
                                 'kv h n d -> kv b h n d',
                                  b = batch)
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)

        k = torch.cat((pmk, k), dim = -2)
        v = torch.cat((pmv, v), dim = -2)

        if not titans_utils.exists(flex_attn_fn):
            block_mask = titans_utils.create_mac_block_mask(seq_len,
                                                            self.total_segment_len,
                                                            self.num_persist_mem_tokens,
                                                            self.sliding)
            flex_attention = torch.compile(flex_attention)
            flex_attn_fn = partial(flex_attention, block_mask = block_mask)

        out = flex_attn_fn(q, k, v)
        out = self.merge_heads(out)
        out = self.to_out(out)
        if titans_utils.exists(output_gating):
            out = out * output_gating
        
        return out, AttnIntermediates(orig_v, next_cache)
    
    def forward(
        self,
        seq,
        value_residual = None,
        flex_atten_fn: Callable | None = None,
        disable_felx_attn = False,
        output_gating = None,
        cache = None,
    ):
        is_inferencing = titans_utils.exists(cache)
        
        if is_inferencing:
            assert seq.shape[-2] == 1
            return self.forward_infernce(seq,
                                         cache,
                                         value_residual, 
                                         output_gating 
                                        )
        
        if seq.is_cuda and self.use_flex_attn and not disable_felx_attn:
            return self.forward_flex(seq, value_residual, flex_atten_fn, output_gating, cache)
        
        assert not (titans_utils.exists(value_residual) ^ titans_utils.exists(self.to_learned_v_mix))

        segment_len, num_longterm_mem_tokens = self.segment_len, self.num_longterm_mem_tokens
        total_segment_len = segment_len + num_longterm_mem_tokens

        batch, seq_len = seq.shape[:2]

        seq, inverse_segment = titans_utils.pad_and_segment_with_inverse(seq, total_segment_len, fold_into_bacth = False)

        seq = self.norm(seq)

        q, k, v = self.to_qkv(seq).chunk(3, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))

        orig_v = v

        if titans_utils.exists(self.to_learned_v_mix):
            mix = self.to_learned_v_mix(seq)
            v = v.lerp(value_residual, mix)
        
        next_cache = tuple(map(inverse_segment, (k ,v)))
        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        q, k, v = tuple(einops.rearrange(t, 'b h (w n) d -> (b w) h n d', n = total_segment_len) for t in (q, k, v))

        attend_kwargs = dict()

        if self.sliding:
            k, v = tuple(einops.rearrange(t, '(b w) ... -> b w ...', b = batch) for t in (k, v))
            k, v = tuple(titans_utils.pad_at_dim(t, (1, 0), value = 0., dim = 1) for t in (k, v))
            k = cat((k[:, :-1], k[:, 1:]), dim = -2)
            v = cat((v[:, :-1], v[:, 1:]), dim = -2)
            k, v = tuple(einops.rearrange(t, 'b w ... -> (b w) ...') for t in (k, v))

            idx = torch.arange(seq.shape[-2], device = seq.device)
            q_idx = einops.rearrange(idx, '(w n) -> w n', n = total_segment_len)
            k_idx = titans_utils.pad_at_dim(q_idx, (1, 0), dim = 0, value = -1e4)
            k_idx = cat((k_idx[:-1], k_idx[1:]), dim = -1)

            q_idx = einops.rearrange(q_idx, 'w i -> w i 1')
            k_idx = einops.rearrange(k_idx, 'w j -> w 1 j')

            sliding_mask = (q_idx - k_idx) <= total_segment_len
            sliding_mask = F.pad(sliding_mask, (self.num_persist_mem_tokens, 0), value = True)

            sliding_mask = einops.repeat(sliding_mask, 'w i j -> (b w) 1 i j', b = batch)
            attend_kwargs.update(mask = sliding_mask)

        pmk, pmv = einops.repeat(self.persistent_memory, 'kv ... -> kv b ...', b = k.shape[0])

        k = cat((pmk, k), dim = -2)
        v = cat((pmv, v), dim = -2)
        out, _ = self.attend(q, k, v, **attend_kwargs)
        out = self.merge_heads(out)
        out = self.to_out(out)
        out = einops.rearrange(out, '(b w) n d -> b (w n) d', b = batch)
        out = inverse_segment(out)

        if titans_utils.exists(output_gating):
            out = out * output_gating

        return out, AttnIntermediates(orig_v, next_cache)
