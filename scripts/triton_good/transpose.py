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