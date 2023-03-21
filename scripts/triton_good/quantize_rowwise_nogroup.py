import math
import torch
import time
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time

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
def _quantize_rowwise_nogroup(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    row_mask = arange < BLOCK_SIZE
    x = tl.load(x_ptr + offsets, mask=row_mask)
    
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, 0), axis=0)
    output = tl.libdevice.llrint(127. * (x / max_val))
    tl.store(output_ptr + offsets, output, mask=row_mask)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise_nogroup(x: torch.Tensor):
    output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device='cuda', dtype=torch.float16)

    P2 = int(2 ** (math.ceil(math.log2(x.shape[1]))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (x.shape[0],)
    _quantize_rowwise_nogroup[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=x.shape[1], P2=P2)
    return output, output_maxs



if __name__ == '__main__':
    torch.manual_seed(0)

    x = torch.randn(1280, 768).cuda().to(torch.float16)
    out = quantize_rowwise_nogroup(x)

    x_real = (127 * x.float() / x.abs().max(dim=1, keepdim=True)[0]).round().to(torch.int8)
    max2 = x.abs().max(1)[0]

    print(torch.allclose(out[1], max2))
    print( (x_real == out[0]).float().mean() )

    # for i in range(x.shape[0]):
    #     print( (x_real[i, :] == out[0][i, :]).float().mean() )

    print(out[0])
    print(x_real)
    exit()

    # repeat = 16

    # for _ in range(8):
    #     out = quantize_rowwise_nogroup(x)

    # triton_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(triton_graph):
    #     out = quantize_rowwise_nogroup(x)

    # triton_graph.replay()

    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     triton_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()

    # print(out[0])
    # print(out[1])
    # print(x / x.abs().max(dim=1, keepdim=True)[0])
    # max1 = out[1]
    # max2 = x.abs().max(1)[0]
    # print(max1, max2)
    # print(torch.allclose(max1, max2))

    #print(f"time: {(end - start) / repeat * 1000:.3f} ms")
