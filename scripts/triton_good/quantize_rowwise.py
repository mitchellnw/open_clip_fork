import math
import torch
import time
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

# TODO: autotune this better.
@triton.autotune(
        configs=[
            # triton.Config({'BLOCK_SIZE_M' : 1}, num_warps=4),
            # triton.Config({'BLOCK_SIZE_M' : 2}, num_warps=4),
            triton.Config({'BLOCK_SIZE_M' : 4}, num_warps=4),
            # triton.Config({'BLOCK_SIZE_M' : 8}, num_warps=4),
            # triton.Config({'BLOCK_SIZE_M' : 16}, num_warps=4),
            # triton.Config({'BLOCK_SIZE_M' : 32}, num_warps=4),
            # triton.Config({'BLOCK_SIZE_M' : 64}, num_warps=4),
        ],
        key=['M', 'N']
)
@triton.jit
def _quantize_rowwise(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    M : tl.constexpr, N: tl.constexpr,
    P2: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * N * BLOCK_SIZE_M
    arange_x = tl.arange(0, P2)
    arange_y = tl.arange(0, BLOCK_SIZE_M)
    offsets = block_start + arange_x[:, None] + N * arange_y[None, :]
    mask = (offsets < (block_start + (N * tl.arange(1, BLOCK_SIZE_M + 1))[None, :])) and offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=offsets < n_elements)
    maxs = tl.max(tl.where(mask, tl.abs(x), 0), axis=0)
    scaled_x =127 * x / maxs[None, :]
    int8_x = scaled_x.to(tl.int8)
    tl.store(output_ptr + offsets, int8_x, mask=mask)
    tl.store(output_maxs + pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M), maxs)

def quantize_rowwise(x: torch.Tensor):
    output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device='cuda', dtype=torch.float16)

    M, N = x.shape
    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
    _quantize_rowwise[grid](x, output, output_maxs, n_elements, M, N, P2=P2)
    return output, output_maxs


if __name__ == '__main__':

    x = torch.randn(256*32, 1280).cuda().to(torch.float16)
    out = quantize_rowwise(x)

    repeat = 16

    for _ in range(8):
        out = quantize_rowwise(x)

    triton_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_graph):
        out = quantize_rowwise(x)

    triton_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    print(out[0])
    print(out[1])
    print(x / x.abs().max(dim=1, keepdim=True)[0])
    max1 = out[1]
    max2 = x.abs().max(1)[0]
    print(max1, max2)
    print(torch.allclose(max1, max2))

    print(f"time: {(end - start) / repeat * 1000:.3f} ms")
