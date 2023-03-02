from itertools import repeat
import collections.abc

from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d


def freeze_batch_norm_2d(module, module_match={}, name=''):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = '.'.join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = lambda n, x: _ntuple(n)(x)


# TODO: replace with timm.. this is from timm.
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    

# from typing import Optional, Tuple
# import torch
# from torch.nn.parameter import Parameter
# from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
# from torch.nn.init import constant_, xavier_normal_, xavier_uniform_
# from torch import Tensor

# class ExtraLNAttention(nn.Module):
#     __constants__ = ['batch_first']
#     bias_k: Optional[torch.Tensor]
#     bias_v: Optional[torch.Tensor]

#     def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
#                  kdim=None, vdim=None, batch_first=False, device=None, dtype=None) -> None:
#         factory_kwargs = {'device': device, 'dtype': dtype}
#         super(ExtraLNAttention, self).__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#         self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

#         self.num_heads = num_heads
#         self.dropout = dropout
#         self.batch_first = batch_first
#         self.head_dim = embed_dim // num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         if self._qkv_same_embed_dim is False:
#             self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
#             self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
#             self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
#             self.register_parameter('in_proj_weight', None)
#         else:
#             self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
#             self.register_parameter('q_proj_weight', None)
#             self.register_parameter('k_proj_weight', None)
#             self.register_parameter('v_proj_weight', None)

#         if bias:
#             self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
#         else:
#             self.register_parameter('in_proj_bias', None)
#         self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

#         if add_bias_kv:
#             self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#             self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
#         else:
#             self.bias_k = self.bias_v = None

#         self.add_zero_attn = add_zero_attn

#         self._reset_parameters()

#     def _reset_parameters(self):
#         if self._qkv_same_embed_dim:
#             xavier_uniform_(self.in_proj_weight)
#         else:
#             xavier_uniform_(self.q_proj_weight)
#             xavier_uniform_(self.k_proj_weight)
#             xavier_uniform_(self.v_proj_weight)

#         if self.in_proj_bias is not None:
#             constant_(self.in_proj_bias, 0.)
#             constant_(self.out_proj.bias, 0.)
#         if self.bias_k is not None:
#             xavier_normal_(self.bias_k)
#         if self.bias_v is not None:
#             xavier_normal_(self.bias_v)

#     def __setstate__(self, state):
#         # Support loading old MultiheadAttention checkpoints generated by v1.1.0
#         if '_qkv_same_embed_dim' not in state:
#             state['_qkv_same_embed_dim'] = True

#         super(ExtraLNAttention, self).__setstate__(state)

#     def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
#                 need_weights: bool = True, attn_mask: Optional[Tensor] = None,
#                 average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
# 
# 
# 
class ExtraLNAttention(nn.Module):
    pass

# import math
# import torch
# class ExtraLNAttention(nn.Module):
#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             qkv_bias=True,
#             scaled_cosine=False,
#             scale_heads=False,
#             logit_scale_max=math.log(1. / 0.01),
#             attn_drop=0.,
#             proj_drop=0.,
#             extra_ln=False,
#     ):
#         super().__init__()
#         self.scaled_cosine = scaled_cosine
#         self.scale_heads = scale_heads
#         assert dim % num_heads == 0, 'dim should be divisible by num_heads'
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.logit_scale_max = logit_scale_max

#         # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
#         self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
#         if qkv_bias:
#             self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
#         else:
#             self.in_proj_bias = None

#         if self.scaled_cosine:
#             self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
#         else:
#             self.logit_scale = None
#         self.attn_drop = nn.Dropout(attn_drop)
#         if self.scale_heads:
#             self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
#         else:
#             self.head_scale = None
#         self.out_proj = nn.Linear(dim, dim)
#         self.out_drop = nn.Dropout(proj_drop)

#     def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
#         L, N, C = x.shape
#         q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
#         q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
#         k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
#         v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

#         if self.logit_scale is not None:
#             attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
#             logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
#             attn = attn.view(N, self.num_heads, L, L) * logit_scale
#             attn = attn.view(-1, L, L)
#         else:
#             q = q * self.scale
#             attn = torch.bmm(q, k.transpose(-1, -2))

#         if attn_mask is not None:
#             if attn_mask.dtype == torch.bool:
#                 new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
#                 new_attn_mask.masked_fill_(attn_mask, float("-inf"))
#                 attn_mask = new_attn_mask
#             attn += attn_mask

#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)

#         x = torch.bmm(attn, v)
#         if self.head_scale is not None:
#             x = x.view(N, self.num_heads, L, C) * self.head_scale
#             x = x.view(-1, L, C)
#         x = x.transpose(0, 1).reshape(L, N, C)
#         x = self.out_proj(x)
#         x = self.out_drop(x)
#         return x


import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """
    PDB Subclass for debugging multi-processed code
    Suggested in: https://stackoverflow.com/questions/4716533/how-to-attach-debugger-to-a-python-subproccess
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin