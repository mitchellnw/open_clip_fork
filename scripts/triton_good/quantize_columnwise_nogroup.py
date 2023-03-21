import math
import torch
import time
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


######################## CAUTION: NOT TESTED ####################################
######################## CAUTION: NOT TESTED ####################################
######################## CAUTION: NOT TESTED ####################################



# TODO: autotune this better.
@triton.autotune(
        configs=[
            # triton.Config({}, num_warps=1),
            triton.Config({}, num_warps=2),
            # triton.Config({}, num_warps=4),
            # triton.Config({}, num_warps=8),
        ],
        key=['n_elements']
)
@triton.jit
def _quantize_columnwise_nogroup(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    M : tl.constexpr, N : tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid
    p2_arage = tl.arange(0, P2)
    arange = p2_arage * N
    offsets = block_start + arange
    p2_mask = p2_arage < M
    x = tl.load(x_ptr + offsets, mask=p2_mask)
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(p2_mask, abs_x, 0), axis=0)
    output = tl.libdevice.llrint(127. * (x / max_val))
    tl.store(output_ptr + offsets, output, mask=p2_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_columnwise_nogroup(x: torch.Tensor):
    M, N = x.shape
    output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
    output_maxs = torch.empty(x.shape[1], device='cuda', dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[0]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_columnwise_nogroup[grid](x, output, output_maxs, n_elements, M, N, BLOCK_SIZE=x.shape[0], P2=P2)
    return output, output_maxs



if __name__ == '__main__':

    x = torch.randn(256*32, 1280).cuda().to(torch.float16)
    out = quantize_columnwise_nogroup(x)

    repeat = 16

    for _ in range(8):
        out = quantize_columnwise_nogroup(x)

    triton_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_graph):
        out = quantize_columnwise_nogroup(x)

    triton_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    print(out[0])
    print(out[1])
    print(x / x.abs().max(dim=0, keepdim=True)[0])
    x_real = (127 * (x / x.abs().max(dim=0, keepdim=True)[0])).round().to(torch.int8)
    max1 = out[1]
    max2 = x.abs().max(0)[0]
    print(max1, max2)
    import pdb; pdb.set_trace()
    print(torch.allclose(max1, max2))

    print(f"time: {(end - start) / repeat * 1000:.3f} ms")
