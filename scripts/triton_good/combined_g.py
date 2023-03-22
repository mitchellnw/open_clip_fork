import math
import torch
import torch.nn as nn

import torch.profiler

# import bitsandbytes as bnb

from int8_matmul_mixed_dequanitze_stable import int8_matmul_mixed_dequanitze_stable, int8_matmul_mixed_dequanitze_bias
from quantize_global import quantize_global, quantize_global_transpose
from quantize_rowwise_nogroup import quantize_rowwise_nogroup, experimental_quantize_rowwise_nogroup
from int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize
from quantize_columnwise_nogroup_transpose import quantize_columnwise_nogroup_transpose
from int8_matmul_rowwise_dequantize_experimental import int8_matmul_rowwise_dequantize_experimental

from transpose import transpose_triton

import triton.ops.matmul as triton_matmul

class _switchback_global(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, W, bias):
        X_int8, state_X = quantize_rowwise_nogroup(X)
        W_int8, state_W = quantize_global(W)
        ctx.save_for_backward = X, W#, W_int8, state_W
        return int8_matmul_mixed_dequanitze_bias(X_int8, W_int8.t(), state_X, state_W, bias)

    @staticmethod
    def backward(ctx, G):
        #X, W, W_int8, state_W = ctx.save_for_backward
        X, W = ctx.save_for_backward
        G_int8, state_G, grad_bias = quantize_rowwise_nogroup(X)
        #W_int8 = torch.transpose(W_int8, 0, 1).contiguous()
        W_int8, state_W = quantize_global_transpose(W)
        grad_X = int8_matmul_mixed_dequanitze_stable(G_int8, W_int8.t(), state_G, state_W).to(X.dtype)
        grad_W = torch.matmul(G.t(), X.to(G.dtype)).to(W.dtype)
        return grad_X, grad_W, G.sum(dim=0)
    
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

        
class SwitchBackLinearGlobal(nn.Linear):
    def forward(self, x):
        return _switchback_global.apply(x, self.weight, self.bias)
    
class SwitchBackLinear(nn.Linear):
    def forward(self, x):
        return _switchback.apply(x, self.weight, self.bias)
    

def backward(G, X, W):
    G_int8, state_G = quantize_rowwise_nogroup(G)
    W_int8, state_W = quantize_columnwise_nogroup_transpose(W)
    grad_X = int8_matmul_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W).to(X.dtype)
    grad_W = torch.matmul(G.t().to(X.dtype), X).to(W.dtype)
    return grad_X, grad_W, None#G.sum(dim=0)

# def backward(G, X, W):
#     grad_X = torch.matmul(G.to(W.dtype), W.t().contiguous().t()).to(X.dtype)
#     grad_W = torch.matmul(G.t().to(X.dtype), X).to(W.dtype)
#     return grad_X, grad_W, G.sum(dim=0)

import time
if __name__ == '__main__':
    torch.manual_seed(0)
    repeat = 16
    dim=1024

    layers = 4

    standard_linear = []
    for _ in range(layers):
        standard_linear.append(nn.Linear(dim, 4*dim))
        standard_linear.append(nn.Linear(4*dim, dim))

    rowwise_linear = []
    for _ in range(layers):
        rowwise_linear.append(SwitchBackLinear(dim, 4*dim))
        rowwise_linear.append(SwitchBackLinear(4*dim, dim))

    standard_linear = nn.Sequential(*standard_linear).cuda().train()
    rowwise_linear = nn.Sequential(*rowwise_linear).cuda().train()


    # standard_linear = nn.Sequential(*[ nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim) for _ in range(10)]).cuda().train()
    # rowwise_linear = nn.Sequential(*[ SwitchBackLinear(dim, 4*dim), nn.GELU(), SwitchBackLinear(4*dim, dim) for _ in range(10)]).cuda().train()

    x1 = torch.randn(256, 256, dim, dtype=torch.float16).cuda()

    # standard forward 2.97
    # rowwise forward 1.68

    # standard forward + back = 9.5
    # rowwise forward + back = 10.1 ?

    # for _ in range(repeat):
    #     out = rowwise_linear(x1)
    #     (2**16 * out.pow(2).mean()).backward()


    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
    #     record_shapes=True,
    #     profile_memory=True,
    #     with_stack=True
    # ) as profiler:
    #     out = rowwise_linear(x1)
    #     (2**16 * out.pow(2).mean()).backward()
    #     profiler.step()


    # print('DONE PRIFOLE')
    # import pdb; pdb.set_trace()

    #################################################
    for _ in range(repeat // 2):
        with torch.cuda.amp.autocast():
            out = standard_linear(x1)
        (2**16 * out.pow(2).mean()).backward()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        with torch.cuda.amp.autocast():
            out = standard_linear(x1)
        (2**16 * out.pow(2).mean()).backward()

    torch.cuda.synchronize()
    end = time.time()
    print(f"time standard: {(end - start) / repeat * 1000:.3f} ms")
    #################################################

    
    #################################################
    for _ in range(repeat // 2):
        with torch.cuda.amp.autocast():
            out = rowwise_linear(x1)
        (2**16 * out.pow(2).mean()).backward()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        with torch.cuda.amp.autocast():
            out = rowwise_linear(x1)
        (2**16 * out.pow(2).mean()).backward()

    torch.cuda.synchronize()
    end = time.time()
    print(f"time rowwise: {(end - start) / repeat * 1000:.3f} ms")
    #################################################

    exit()



    G = torch.randn(256*256, 4 * dim, dtype=torch.float16).cuda()
    X = torch.randn(256*256, dim, dtype=torch.float16).cuda()
    W = torch.randn(4 * dim, dim, dtype=torch.float32).cuda()
    ##################
    for _ in range(8):
        backward(G, X, W)

    torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_matmul_fp16_graph):
        backward(G, X, W)

    torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"fp16 matmul: {(end - start) / repeat * 1000:.3f} ms")
    matmul_time = (end - start) / repeat
    ##################
