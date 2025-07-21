from typing import Callable
import math
from functools import partial
from collections import namedtuple
from itertools import zip_longest

import torch
from torch import nn
from torch.func import functional_call, vmap, grad
import torch.nn.functional as F
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten
from tensordict import TensorDict

from titans.memory import(
    MemoryMLP,
    ResidualNorm
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

