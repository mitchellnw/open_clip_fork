import math
import torch
import torch.nn as nn

import bitsandbytes as bnb

from int8_matmul_mixed_dequanitze_stable import int8_matmul_mixed_dequanitze_stable
from quantize_global import quantize_global
from quantize_rowwise_nogroup import quantize_rowwise_nogroup
from int8_matmul_rowwise_dequantize import int8_matmul_rowwise_dequantize
from quantize_columnwise_nogroup_transpose import quantize_columnwise_nogroup_transpose
from transpose import transpose_triton

import triton.ops.matmul as triton_matmul

class _switchback_global(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, W):
        X_int8, state_X = quantize_rowwise_nogroup(X)
        W_int8, state_W = quantize_global(W)
        ctx.save_for_backward = X, W, W_int8, state_W
        return int8_matmul_mixed_dequanitze_stable(X_int8, W_int8.t(), state_X, state_W)

    @staticmethod
    def backward(ctx, G):
        X, W, W_int8, state_W = ctx.save_for_backward
        G_int8, state_G = quantize_rowwise_nogroup(G)
        W_int8 = torch.transpose(W_int8, 0, 1).contiguous()
        grad_X = int8_matmul_mixed_dequanitze_stable(G_int8, W_int8.t(), state_G, state_W).to(X.dtype)
        grad_W = torch.matmul(G.t(), X.to(G.dtype)).to(W.dtype)
        return grad_X, grad_W
    
class _switchback(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, W):
        ctx.save_for_backward = X, W
        X_int8, state_X = quantize_rowwise_nogroup(X)
        W_int8, state_W = quantize_rowwise_nogroup(W)
        return int8_matmul_rowwise_dequantize(X_int8, W_int8.t(), state_X, state_W)
    
    @staticmethod
    def backward(ctx, G):
        X, W = ctx.save_for_backward
        G_int8, state_G = quantize_rowwise_nogroup(G)
        W_int8, state_W = quantize_columnwise_nogroup_transpose(W)
        grad_X = int8_matmul_rowwise_dequantize(G_int8, W_int8.t(), state_G, state_W).to(X.dtype)
        grad_W = torch.matmul(G.t(), X.to(G.dtype)).to(W.dtype)
        return grad_X, grad_W
    
class SwitchBackLinearGlobal(nn.Linear):
    def forward(self, x):
        return _switchback_global.apply(x, self.weight) + self.bias
    
class SwitchBackLinear(nn.Linear):
    def forward(self, x):
        return _switchback.apply(x, self.weight) + self.bias
    

import time
if __name__ == '__main__':
    torch.manual_seed(0)
    repeat = 16
    dim=1024

    # X = torch.randn(256 * 64, 1024, device='cuda', dtype=torch.float16)
    # W = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

    # out = torch.matmul(X, W)
    # start = time.time()
    # out = torch.matmul(X, W)
    # end = time.time() - start

    # print('fp16 time', end)

    # ####################
    # for _ in range(8):
    #     th_c = torch.matmul(X, W)

    # torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(torch_matmul_fp16_graph):
    #     th_c = torch.matmul(X, W)

    # torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"torch fp16 matmul: {(end - start) / repeat * 1000:.3f} ms")
    # fp16_time = (end - start) / repeat
    # ####################

    # X_int8, state_X = quantize_rowwise_nogroup(X)
    # W_int8, state_W = quantize_columnwise_nogroup_transpose(W)
    # W_int8 = W_int8.t()
    # ####################
    # for _ in range(8):
    #     th_c = int8_matmul_mixed_dequanitze_stable(X_int8, W_int8, state_X, state_W)

    # torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(torch_matmul_fp16_graph):
    #     th_c = int8_matmul_mixed_dequanitze_stable(X_int8, W_int8, state_X, state_W)

    # torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     torch_matmul_fp16_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()
    # print(f"triton matmul: {(end - start) / repeat * 1000:.3f} ms")
    # matmul_time = (end - start) / repeat
    # ####################


    fp16_linear = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim)).cuda()

    rowwise_linear = nn.Sequential(SwitchBackLinear(dim, 4*dim), nn.GELU(), SwitchBackLinear(4*dim, dim)).cuda()
    rowwise_linear[0].weight.data = fp16_linear[0].weight.data
    rowwise_linear[0].bias.data = fp16_linear[0].bias.data
    rowwise_linear[2].weight.data = fp16_linear[2].weight.data
    rowwise_linear[2].bias.data = fp16_linear[2].bias.data

    global_linear = nn.Sequential(SwitchBackLinearGlobal(dim, 4*dim), nn.GELU(), SwitchBackLinearGlobal(4*dim, dim)).cuda()
    global_linear[0].weight.data = fp16_linear[0].weight.data
    global_linear[0].bias.data = fp16_linear[0].bias.data
    global_linear[2].weight.data = fp16_linear[2].weight.data
    global_linear[2].bias.data = fp16_linear[2].bias.data

    bnb_linear = nn.Sequential(bnb.nn.Linear8bitLtMixed(dim, 4*dim), nn.GELU(), bnb.nn.Linear8bitLtMixed(4*dim, dim)).cuda()
    bnb_linear[0].weight.data = fp16_linear[0].weight.data
    bnb_linear[0].bias.data = fp16_linear[0].bias.data
    bnb_linear[2].weight.data = fp16_linear[2].weight.data
    bnb_linear[2].bias.data = fp16_linear[2].bias.data

    x1 = torch.randn(256*256, dim, dtype=torch.float16).cuda().requires_grad_(True)
    x2 = x1.clone().detach().requires_grad_(True)
    x3 = x1.clone().detach().requires_grad_(True)
    w_new = torch.randn(dim, 1, dtype=torch.float16).cuda()

    #with torch.cuda.amp.autocast(dtype=torch.float16):
    out_fp16 = fp16_linear(x1.float())
    out_rowwise = rowwise_linear(x2)
    out_global = global_linear(x2)
    out_bnb = bnb_linear(x3)

    #import pdb; pdb.set_trace()

    err_rowwise = (out_rowwise.float() - out_fp16.float()).abs().mean()
    err_global = (out_global.float() - out_fp16.float()).abs().mean()
    err_bnb = (out_bnb.float() - out_fp16.float()).abs().mean()

    print(err_rowwise, err_global, err_bnb)

    out_fp16.mean().backward()
    out_rowwise.mean().backward()
    out_global.mean().backward()
    out_bnb.mean().backward()

    err_rowwise = (rowwise_linear[0].weight.grad - fp16_linear[0].weight.grad).abs().mean()
    err_global = (global_linear[0].weight.grad - fp16_linear[0].weight.grad).abs().mean()
    err_bnb = (bnb_linear[0].weight.grad - fp16_linear[0].weight.grad).abs().mean()



    # err_rowwise = (out_rowwise - out_fp16).abs().mean()
    # err_global = (out_global - out_fp16).abs().mean()
    # err_bnb = (out_bnb - out_fp16).abs().mean()
    
    print(err_rowwise, err_global, err_bnb)


    ##################
    for _ in range(8):
        with torch.cuda.amp.autocast():
            out = fp16_linear(x1)

    torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_matmul_fp16_graph):
        with torch.cuda.amp.autocast():
            out = fp16_linear(x1)

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


    ##################
    for _ in range(8):
        out = global_linear(x1)
        

    torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_matmul_fp16_graph):
        out = global_linear(x1)

    torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"global matmul: {(end - start) / repeat * 1000:.3f} ms")
    matmul_time = (end - start) / repeat
    ##################

    ##################
    for _ in range(8):
        out = rowwise_linear(x1)

    torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_matmul_fp16_graph):
        out = rowwise_linear(x1)

    torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"rowwise matmul: {(end - start) / repeat * 1000:.3f} ms")
    matmul_time = (end - start) / repeat
    ##################


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
