import math
import torch
import torch.nn as nn

import bitsandbytes as bnb

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
        G_int8, state_G, grad_bias = experimental_quantize_rowwise_nogroup(G)
        #W_int8 = torch.transpose(W_int8, 0, 1).contiguous()
        W_int8, state_W = quantize_global_transpose(W)
        grad_X = int8_matmul_mixed_dequanitze_stable(G_int8, W_int8.t(), state_G, state_W).to(X.dtype)
        grad_W = torch.matmul(G.t(), X.to(G.dtype)).to(W.dtype)
        return grad_X, grad_W, grad_bias
    
class _switchback(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, W, bias):
        ctx.save_for_backward = X, W
        X_int8, state_X = quantize_rowwise_nogroup(X)
        W_int8, state_W = quantize_rowwise_nogroup(W)
        return int8_matmul_rowwise_dequantize_experimental(X_int8, W_int8.t(), state_X, state_W, bias)
    
    @staticmethod
    def backward(ctx, G):
        X, W = ctx.save_for_backward
        G_int8, state_G, grad_bias = experimental_quantize_rowwise_nogroup(G)
        W_int8, state_W = quantize_columnwise_nogroup_transpose(W)
        grad_X = int8_matmul_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W).to(X.dtype)
        grad_W = torch.matmul(G.t(), X.to(G.dtype)).to(W.dtype)
        return grad_X, grad_W, grad_bias
    
class SwitchBackLinearGlobal(nn.Linear):
    def forward(self, x):
        return _switchback_global.apply(x, self.weight, self.bias)
    
class SwitchBackLinear(nn.Linear):
    def forward(self, x):
        return _switchback.apply(x, self.weight, self.bias)
    


import time
if __name__ == '__main__':
    torch.manual_seed(0)
    repeat = 16
    dim=2048


    fp16_linear = nn.Linear(dim, 4*dim, bias=None).cuda()

    rowwise_linear = SwitchBackLinear(dim, 4*dim, bias=None).cuda()
    rowwise_linear.weight.data = fp16_linear.weight.data.clone()
    #rowwise_linear.bias.data = fp16_linear.bias.data.clone()

    global_linear = SwitchBackLinearGlobal(dim, 4*dim).cuda()
    global_linear.weight.data = fp16_linear.weight.data.clone()
    #global_linear.bias.data = fp16_linear.bias.data.clone()

    x1 = torch.randn(256*256, dim, dtype=torch.float16).cuda()
    x2 = x1.clone().detach()
    x3 = x1.clone().detach()

    X = torch.randn(256*256, dim, dtype=torch.float16).cuda()
    G = torch.randn(256*256, dim*4, dtype=torch.float16).cuda()
    W = torch.randn(4 * dim, dim, dtype=torch.float16).cuda()


    ##################
    for _ in range(8):
        out = torch.matmul(G.t(), X)

    torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_matmul_fp16_graph):
        out = torch.matmul(G.t(), X)

    torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"fp16 matmul 1: {(end - start) / repeat * 1000:.3f} ms")
    matmul_time = (end - start) / repeat
    ##################


    ##################
    for _ in range(8):
        out = torch.matmul(X, W.t())

    torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_matmul_fp16_graph):
        out = torch.matmul(X, W.t())

    torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"fp16 matmul 2: {(end - start) / repeat * 1000:.3f} ms")
    matmul_time = (end - start) / repeat
    ##################
    
    # ##################
    # for _ in range(8):
    #     X_int8, statex = quantize_rowwise_nogroup(X)

    # torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(torch_matmul_fp16_graph):
    #     X_int8, statex = quantize_rowwise_nogroup(X)

    # torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"quantize x time: {(end - start) / repeat * 1000:.3f} ms")
    # matmul_time = (end - start) / repeat
    # ##################

    # ##################
    # for _ in range(8):
    #     W_int8, statew = quantize_rowwise_nogroup(W)

    # torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(torch_matmul_fp16_graph):
    #     W_int8, statew = quantize_rowwise_nogroup(W)

    # torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"quantize w time: {(end - start) / repeat * 1000:.3f} ms")
    # matmul_time = (end - start) / repeat
    # ##################

    # ##################
    # for _ in range(8):
    #     out = int8_matmul_rowwise_dequantize(X_int8, W_int8.t(), statex, statew)

    # torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(torch_matmul_fp16_graph):
    #     out = int8_matmul_rowwise_dequantize(X_int8, W_int8.t(), statex, statew)

    # torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"int8 matmul: {(end - start) / repeat * 1000:.3f} ms")
    # matmul_time = (end - start) / repeat
    # ##################


    ##################
    # for _ in range(8):
    #     with torch.cuda.amp.autocast():
    #         out = fp16_linear(x1)

    # torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(torch_matmul_fp16_graph):
    #     with torch.cuda.amp.autocast():
    #         out = fp16_linear(x1)

    # torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"fp16 matmul: {(end - start) / repeat * 1000:.3f} ms")
    # matmul_time = (end - start) / repeat
    ##################


    # ##################
    # for _ in range(8):
    #     with torch.cuda.amp.autocast():
    #         out = global_linear(x1)
        

    # torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(torch_matmul_fp16_graph):
    #     with torch.cuda.amp.autocast():
    #         out = global_linear(x1)

    # torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"global matmul: {(end - start) / repeat * 1000:.3f} ms")
    # matmul_time = (end - start) / repeat
    # ##################

    ##################
    # for _ in range(8):
    #     out_rowwise = rowwise_linear(x1)

    # torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(torch_matmul_fp16_graph):
    #     out_rowwise = rowwise_linear(x1)

    # torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"rowwise matmul: {(end - start) / repeat * 1000:.3f} ms")
    # matmul_time = (end - start) / repeat
    ##################

    #print((out_rowwise - out).abs().mean())


    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     with torch.cuda.amp.autocast():
    #         out = fp16_linear(x1)
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"fp16 matmul: {(end - start) / repeat * 1000:.3f} ms")


    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     out = global_linear(x2)
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"global matmul: {(end - start) / repeat * 1000:.3f} ms")

    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     out = rowwise_linear(x3)
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"rowwise matmul: {(end - start) / repeat * 1000:.3f} ms")




# class _pytorch_int8(torch.autograd.Function):
#     def forward(ctx, X, W):
#         ctx.save_for_backward = X, W
#         state_X = X.abs().max(dim=1, keepdim=True)[0]
#         X_int8 = ((127 * X) / state_X).round()
#         Wt = W.t()
#         state_W = Wt.abs().max(dim=0, keepdim=True)[0]
#         W_int8 = ((127 * Wt) / state_W).round()
#         X_fp16 = state_X * X_int8 / 127
#         W_fp16 = state_W * W_int8 / 127
#         return torch.matmul(X_fp16, W_fp16)

    # def backward(ctx, G):
    #     X, W = ctx.save_for_backward
    #     state_G = G.abs().max(dim=1, keepdim=True)[0]
    #     G_int8 = ((127 * G) / state_G).round()
    #     state_W = W.abs().max(dim=0, keepdim=True)[0]
    #     W_int8 = ((127 * W) / state_W).round()
    #     G_fp16 = state_G * G_int8 / 127
    #     W_fp16 = state_W * W_int8 / 127
    #     grad_X = torch.matmul(G_fp16, W_fp16)
    #     grad_W = torch.matmul(G.t(), X.to(G.dtype)).to(W.dtype)
    #     return grad_X, grad_W
