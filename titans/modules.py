import torch
from torch import nn, cat
import torch.nn.functional as F
from utils import LayerNorm, l2norm
import titans.utils as titans_utils
import einops
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
    