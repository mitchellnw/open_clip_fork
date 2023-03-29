import math
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


import torch.profiler

# import bitsandbytes as bnb

import torchvision
# import open_clip_lr
#import tkernels

from tritongood.int8_matmul_mixed_dequanitze_stable import int8_matmul_mixed_dequanitze_stable, int8_matmul_mixed_dequanitze_bias
from tritongood.quantize_global import quantize_global, quantize_global_transpose
from tritongood.quantize_rowwise_nogroup import quantize_rowwise_nogroup, experimental_quantize_rowwise_nogroup
from tritongood.int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize
from tritongood.quantize_columnwise_nogroup_transpose import quantize_columnwise_nogroup_transpose
from tritongood.int8_matmul_rowwise_dequantize_experimental import int8_matmul_rowwise_dequantize_experimental

from tritongood.transpose import transpose_triton

# import triton.ops.matmul as triton_matmul

#import open_clip
# from open_clip.loss import ClipLoss

#import os
def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size

def init_distributed_device():
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    rank = 0  # global rank
    local_rank = 0

    # DDP via torchrun, torch.distributed.launch
    local_rank, _, _ = world_info_from_env()
    torch.distributed.init_process_group(
        backend='nccl',
        init_method="env://")
    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()
    distributed = True

    if torch.cuda.is_available():
        device = 'cuda:%d' % local_rank
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    device = torch.device(device)
    return device


class _switchback(torch.autograd.Function):

    # @staticmethod
    # def forward(ctx, X_3D, W, bias):

    #     X = X_3D.view(-1, X_3D.size(-1))

    #     ctx.save_for_backward = X, W
    #     print('h1')
    #     X_int8, state_X = quantize_rowwise_nogroup(X)
    #     print('h2')
    #     W_int8, state_W = quantize_rowwise_nogroup(W)
    #     print('h3')
    #     out = int8_matmul_rowwise_dequantize_experimental(
    #         X_int8, W_int8.t(), state_X, state_W, bias
    #     ).view(*X_3D.size()[:-1], -1)
    #     print('h4')
    @staticmethod
    def forward(ctx, X_3D, W, bias):

        X = X_3D.view(-1, X_3D.size(-1))

        ctx.save_for_backward = X, W
        X_int8, state_X = quantize_rowwise_nogroup(X)
        W_int8, state_W = quantize_rowwise_nogroup(W)
        return int8_matmul_rowwise_dequantize_experimental(
            X_int8, W_int8.t(), state_X, state_W, bias
        ).view(*X_3D.size()[:-1], -1)
    
    @staticmethod
    def backward(ctx, G_3D):
        X, W = ctx.save_for_backward

        G = G_3D.view(-1, G_3D.size(-1))

        grad_X = grad_W = grad_bias = None

        if ctx.needs_input_grad[0]:
            G_int8, state_G = quantize_rowwise_nogroup(G)
            W_int8, state_W = quantize_columnwise_nogroup_transpose(W)
            grad_X = int8_matmul_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W).view(
                *G_3D.size()[:-1], -1
            )
        if ctx.needs_input_grad[1]:
            grad_W = torch.matmul(G.t(), X)
        if ctx.needs_input_grad[2]:
            grad_bias = G.sum(dim=0)

        return grad_X, grad_W, grad_bias
    

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        output = input.matmul(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class SwitchBackLinear(nn.Linear):
    def forward(self, x):
        return _switchback.apply(x, self.weight, self.bias)

import time
if __name__ == '__main__':
    #torch.manual_seed(0)
    repeat = 1
    dim=768

    layers = 1

    device = 'cuda:0'#init_distributed_device()

    rowwise_linear = []
    for _ in range(layers):
        rowwise_linear.append(SwitchBackLinear(dim, 4*dim))
        rowwise_linear.append(SwitchBackLinear(4*dim, dim))
    rowwise_linear = nn.Sequential(*rowwise_linear).cuda().train()

    # rowwise_linear = []
    # for _ in range(layers):
    #     rowwise_linear.append(SwitchBackLinear(dim, 4*dim))
    #     rowwise_linear.append(nn.GELU())
    #     rowwise_linear.append(SwitchBackLinear(4*dim, dim))

    #standard_linear = nn.Sequential(*standard_linear).cuda().train()
    #rowwise_linear = nn.Sequential(*rowwise_linear).cuda()
    ##rowwise_linear = SwitchBackLinear(dim, dim).cuda().train()
    #rowwise_linear = DDP(rowwise_linear, device_ids=[device])

    # for _ in range(2):

    #     print(device)


    #fake_data = torch.randn(256 * 32, dim, dtype=torch.float16).cuda()
    x1 = torch.randn(10, 10, dim, dtype=torch.float16).cuda()
    #with torch.cuda.amp.autocast():
    out = rowwise_linear(x1)

    print('its fine')

    exit()


