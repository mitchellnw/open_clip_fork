import torch

import triton
import triton.language as tl

@triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4),
            triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'GROUP_M': 8}, num_warps=4),

            # ...
        ],
        key=['M', 'N']
)
@triton.jit
def _transpose_triton(A, B, stride_am, stride_an, stride_bn, stride_bm, M, N, 
                      BLOCK_M : tl.constexpr, 
                      BLOCK_N : tl.constexpr, 
                      GROUP_M : tl.constexpr):
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    A = A + (rm[:, None] * stride_am + rn[None, :] * stride_an)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    a = tl.load(A, mask=mask)
    
    # rematerialize to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    B = B + (rm[:, None] * stride_bm + rn[None, :] * stride_bn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]
    tl.store(B, a, mask=mask)

def transpose_triton(input, out=None):
    M, N = input.shape
    if out is None:
        out = input.new_zeros(N, M)
    
    assert out.size(0) == N and out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert out.stride(0) == 1 or out.stride(1) == 1
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    _transpose_triton[grid](input, out, input.stride(0), input.stride(1), out.stride(0), out.stride(1), M, N)
    return out

@triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 1,}, num_warps=4),
            triton.Config({'BLOCK_M': 2,}, num_warps=4),
            triton.Config({'BLOCK_M': 4,}, num_warps=4),
            triton.Config({'BLOCK_M': 8,}, num_warps=4),
            triton.Config({'BLOCK_M': 16,}, num_warps=4),
            triton.Config({'BLOCK_M': 32,}, num_warps=4),
            triton.Config({'BLOCK_M': 64,}, num_warps=4),
        ],
        key=['M', 'N']
)
@triton.jit
def _transpose_triton_v2(A, B, stride_am, stride_an, stride_bn, stride_bm, 
                      M : tl.constexpr, N : tl.constexpr, 
                      BLOCK_M : tl.constexpr):
    # pid = tl.program_id(0)
    # grid_in = pid * N + tl.arange(0, N)
    # grid_out = pid + M * tl.arange(0, N)
    # info = tl.load(A + grid_in)
    # tl.store(B + grid_out, info)

    pid = tl.program_id(0)
    grid_in_x = tl.arange(0, N)
    grid_in_y = N * tl.arange(0, BLOCK_M)
    grid_in = pid * N * BLOCK_M + (grid_in_x[:, None] + grid_in_y[None, :])

    grid_out_x = M * tl.arange(0, N)
    grid_out_y = tl.arange(0, BLOCK_M)
    grid_out = pid * BLOCK_M + (grid_out_x[:, None] + grid_out_y[None, :])
    info = tl.load(A + grid_in)
    tl.store(B + grid_out, info)

def transpose_triton_v2(input, out=None):
    M, N = input.shape
    if out is None:
        out = input.new_zeros(N, M)
    
    assert out.size(0) == N and out.size(1) == M
    assert input.stride(0) == 1 or input.stride(1) == 1
    assert out.stride(0) == 1 or out.stride(1) == 1
    
    #grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    _transpose_triton_v2[grid](input, out, input.stride(0), input.stride(1), out.stride(0), out.stride(1), M, N)
    return out


@triton.autotune(
        configs=[
            triton.Config({'BLOCK_M': 1,}, num_warps=4),
            triton.Config({'BLOCK_M': 2,}, num_warps=4),
            triton.Config({'BLOCK_M': 4,}, num_warps=4),
            triton.Config({'BLOCK_M': 8,}, num_warps=4),
            triton.Config({'BLOCK_M': 16,}, num_warps=4),
            triton.Config({'BLOCK_M': 32,}, num_warps=4),
            triton.Config({'BLOCK_M': 64,}, num_warps=4),
        ],
        key=['M', 'N']
)
@triton.jit
def _transpose_and_global_quantize(A, absmax, B, stride_am, stride_an, stride_bn, stride_bm, 
                      M : tl.constexpr, N : tl.constexpr, 
                      BLOCK_M : tl.constexpr):

    pid = tl.program_id(0)
    grid_in_x = tl.arange(0, N)
    grid_in_y = N * tl.arange(0, BLOCK_M)
    grid_in = pid * N * BLOCK_M + (grid_in_x[:, None] + grid_in_y[None, :])

    grid_out_x = M * tl.arange(0, N)
    grid_out_y = tl.arange(0, BLOCK_M)
    grid_out = pid * BLOCK_M + (grid_out_x[:, None] + grid_out_y[None, :])
    info = tl.load(A + grid_in)
    absmax_val = tl.load(absmax)
    info_int8 = (127  * info / absmax_val).to(tl.int8)
    tl.store(B + grid_out, info_int8)

def transpose_and_global_quantize(input, out=None):
    M, N = input.shape
    if out is None:
        out = torch.empty(N, M, device=input.device, dtype=torch.int8)

    absmax = input.abs().max().unsqueeze(0)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']),)
    _transpose_and_global_quantize[grid](input, absmax, out, input.stride(0), input.stride(1), out.stride(0), out.stride(1), M, N)
    return out, absmax


import time
def test():
    repeat = 16
    A = torch.randn(2048, 2048, device='cuda', dtype=torch.float32)


    ###########################
    # for _ in range(8):
    #     out = transpose_triton(A)

    # triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(triton_matmul_int8_graph):
    #     out = transpose_triton(A)

    # triton_matmul_int8_graph.replay()

    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     triton_matmul_int8_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()

    # print(f"triton v1: {(end - start) / repeat * 1000:.3f} ms")
    ###########################


    ###########################
    for _ in range(8):
        out, absmax = transpose_and_global_quantize(A)

    triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_matmul_int8_graph):
        out, absmax = transpose_and_global_quantize(A)

    triton_matmul_int8_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_matmul_int8_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    print(f"triton v2: {(end - start) / repeat * 1000:.3f} ms")

    # ###########################
    # for _ in range(8):
    #     out = transpose_triton_v2(A)

    # triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(triton_matmul_int8_graph):
    #     out = transpose_triton_v2(A)

    # triton_matmul_int8_graph.replay()

    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     triton_matmul_int8_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()

    # print(f"triton v2: {(end - start) / repeat * 1000:.3f} ms")


    ##########################
    for _ in range(8):
        out_torch = A.t().clone()

    triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_matmul_int8_graph):
        out_torch = A.t().clone()

    triton_matmul_int8_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_matmul_int8_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    print(f"torch: {(end - start) / repeat * 1000:.3f} ms")
    ##################################

    print(out.to(torch.float16) * (absmax / 127))
    print(out_torch)