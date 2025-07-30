from __future__ import annotations
from typing import Callable

from math import ceil
from copy import deepcopy
from functools import partial
from collections import namedtuple

import tqdm
import torch
from torch import nn
import torch.nn.functional as F

from einops import repeat, rearrange, pack, unpack, einsum
from einops.layers.torch import Rearrange
from axial_positional_embedding import ContinuousAxialPositionalEmbedding
from rotary_embedding_torch import RotaryEmbedding
from hyper_connections import get_init_and_expand_reduce_stream_functions
from x_transformers.attend import Attend







