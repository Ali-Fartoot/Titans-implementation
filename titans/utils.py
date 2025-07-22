import torch.nn.functional as F
import torch; import torch.nn as nn
from einops import einsum, rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange, Reduce
import math

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.layer_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.gamma = nn.Parameter(torch.zeros(dim))
    
    def forward(self, x):
        gamma = self.gamma

        if gamma.ndim == 2:
            gamma = gamma.unsuqeeze(1)
        
        return self.layer_norm(x) * (gamma + 1.)
    

def exists(v):
    return v is not None

def default(*args):
    for arg in args:
        if exists(arg):
            return arg
    return None

def identity(t):
    return t

def xnor(x, y):
    return not (x ^ y)

def divisible_by(num, den):
    return (num % den) == 0

def safe_cat(inputs, dim = -2):
    inputs = tuple(filter(exists, inputs))

    if len(inputs) == 0:
        return None
    elif len(inputs) == 1:
        return inputs[0]

    return nn.cat(inputs, dim = dim)

def is_empty_tensor(t):
    return t.numel() == 0

def dict_get_value_shapes(td):
    return [v.shape for k, v in td.items()]

def rearrange_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: rearrange(t, pattern, **kwargs))

def repeat_dict_values(td, pattern, **kwargs):
    return td.apply(lambda t: repeat(t, pattern, **kwargs))

def pair(v):
    return (v, v) if not isinstance(v, tuple) else v

def round_down_multiple(seq, mult):
    return seq // mult * mult

def round_up_multiple(seq, mult):
    return math.ceil(seq / mult) * mult

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse

def Sequential(*modules):
    modules = [*filter(exists, modules)]

    if len(modules) == 0:
        return nn.Identity()

    if len(modules) == 1:
        return modules[0]

    return nn.Sequential(*modules)

def l2norm(x):
    return F.normalize(x, dim=-1)


def softclamp_max(t, max_value):
    half_max_value = max_value / 2
    return ((t / half_max_value).tanh() * half_max_value) + half_max_value

def softclamp_grad_norm(t, max_value):
    if is_empty_tensor(t):
        return t

    t, inverse = pack_one_with_inverse(t, 'bn *')

    norm = t.norm(dim = -1, keepdim = True)
    clamped_norm = softclamp_max(norm, max_value)

    t = t * (clamped_norm / norm)
    return inverse(t)

def newtonschulz5(
    t,
    steps = 5,
    eps = 1e-7,
    coefs = (3.4445, -4.7750, 2.0315)
):
    if t.ndim <= 3:
        return t

    shape = t.shape
    should_transpose = shape[-2] > shape[-1]

    if should_transpose:
        t = t.transpose(-1, -2)

    t, inv_pack = pack_one_with_inverse(t, '* i j')
    t = t / t.norm(dim = (-1, -2), keepdim = True).clamp(min = eps)

    a, b, c = coefs

    for _ in range(steps):
        A = t @ t.transpose(-1, -2)
        B = b * A + c * A @ A
        t = a * t + B @ t

    if should_transpose:
        t = t.transpose(-1, -2)

    return inv_pack(t)