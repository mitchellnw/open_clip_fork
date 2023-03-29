import math
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


import torch.profiler

# import bitsandbytes as bnb

from tritongood.int8_matmul_mixed_dequanitze_stable import int8_matmul_mixed_dequanitze_stable, int8_matmul_mixed_dequanitze_bias
from tritongood.quantize_global import quantize_global, quantize_global_transpose
from tritongood.quantize_rowwise_nogroup import quantize_rowwise_nogroup, experimental_quantize_rowwise_nogroup
from tritongood.int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize
from tritongood.quantize_columnwise_nogroup_transpose import quantize_columnwise_nogroup_transpose
from tritongood.int8_matmul_rowwise_dequantize_experimental import int8_matmul_rowwise_dequantize_experimental

from tritongood.transpose import transpose_triton

import triton.ops.matmul as triton_matmul

import open_clip

import os
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


def replace_linear(model, linear_replacement, skip_modules=["lm_head", "conv1", "embedding"], copy_weights=True):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            if name in ['in_proj_linear', 'out_proj', 'c_fc', 'c_proj']:
                old_module = model._modules[name]
                model._modules[name] = linear_replacement(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                )
                
                # if copy_weights:
                #     model._modules[name].weight.data.copy_(old_module.weight.data)
                #     if model._modules[name].bias is not None:
                #         model._modules[name].bias.data.copy_(old_module.bias)

    return model

class _switchback(torch.autograd.Function):

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
    torch.manual_seed(0)
    repeat = 16
    dim=768

    layers = 1

    device = init_distributed_device()

    rowwise_linear = []
    for _ in range(layers):
        rowwise_linear.append(SwitchBackLinear(dim, 4*dim))
        rowwise_linear.append(nn.GELU())
        rowwise_linear.append(SwitchBackLinear(4*dim, dim))

    # #standard_linear = nn.Sequential(*standard_linear).cuda().train()
    rowwise_linear = nn.Sequential(*rowwise_linear).to(device)

    from open_clip import create_model_and_transforms
    #model, _, _ = create_model_and_transforms('ViT-B-32', precision='fp16', device='cpu')
    model = SwitchBackLinear(dim, 4*dim).to(device)
    #model.apply(lambda m : setattr(m, 'advanced_logging', False))
    #replace_linear(model, SwitchBackLinear, skip_modules=["lm_head", "conv1", "embedding"])
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device])

    for _ in range(2):
        # fake_img = torch.randn(32, 3, 224, 224, dtype=torch.float16).to(device)
        # fake_txt = torch.randint(low=0, high=1000, size=(32, 77)).to(device)
        fake_x = torch.randn(32, 32, dim).to(device).half()

        with torch.cuda.amp.autocast():
            out = model(fake_x)
            #out = model(fake_img, fake_txt)

    print('its fine')

    exit()


