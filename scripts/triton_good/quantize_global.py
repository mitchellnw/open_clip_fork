import math
import torch
import time
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

# TODO: autotune this better.
@triton.autotune(
        configs=[
            triton.Config({'BLOCK_SIZE': 1024,}, num_warps=4),
            
        ],
        key=['n_elements']
)
@triton.jit
def _quantize_global(
    x_ptr,
    absmax_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    absmax = tl.load(absmax_ptr)
    out = 127 * x / absmax
    output = out.to(tl.int8)
    tl.store(output_ptr + offsets, output, mask=mask)

def quantize_global(x: torch.Tensor, transpose=True):
    absmax = x.abs().max().unsqueeze(0)
    output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_global[grid](x, absmax, output, n_elements)
    if transpose:
        output = output.t()
    return output, absmax


if __name__ == '__main__':


    w = torch.randn(2048, 2048).cuda()
    out = quantize_global(w, transpose=False)

    repeat = 16

    for _ in range(8):
        out = quantize_global(w, transpose=False)

    triton_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_graph):
        out = quantize_global(w, transpose=False)

    triton_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    print(f"time: {(end - start) / repeat * 1000:.3f} ms")
