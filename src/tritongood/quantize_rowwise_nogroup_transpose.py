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
def _quantize_rowwise_nogroup_transpose(
    x_ptr,
    output_ptr,
    output_maxs,
    M : tl.constexpr, N : tl.constexpr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    P2: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, P2)
    offsets = block_start + arange
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    row_mask = arange < BLOCK_SIZE
    abs_x = tl.abs(x)
    max_val = tl.max(tl.where(row_mask, abs_x, -1), axis=0)
    output = tl.libdevice.llrint(127. * (x / max_val))
    new_offsets =  pid + tl.arange(0, P2) * M
    tl.store(output_ptr + new_offsets, output, mask=new_offsets < N * M)
    tl.store(output_maxs + pid, max_val)

def quantize_rowwise_nogroup_transpose(x: torch.Tensor):
    M, N = x.shape
    output = torch.empty(N, M, device='cuda', dtype=torch.int8)
    output_maxs = torch.empty(M, device='cuda', dtype=torch.float16)
    P2 = int(2 ** (math.ceil(math.log2(N))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _quantize_rowwise_nogroup_transpose[grid](x, output, output_maxs, M, N, n_elements, BLOCK_SIZE=N, P2=P2)
    return output, output_maxs


def pt_round(x):
    return (x + 0.5).floor()
    #return x.round()
    # sign_x = x.sign()
    # pos_x = sign_x * x
    # round_x = (0.5 + pos_x).floor()
    # return round_x * sign_x

if __name__ == '__main__':
    torch.manual_seed(0)

    x = torch.randn(4096, 4096).cuda().to(torch.float16)
    out = quantize_rowwise_nogroup_transpose(x)

    repeat = 16

    for _ in range(8):
        out = quantize_rowwise_nogroup_transpose(x)

    triton_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_graph):
        out = quantize_rowwise_nogroup_transpose(x)

    triton_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    # print(out[0])
    # print(out[1])

    #xt = x.t()
    xt = x.to(torch.float32)
    x_real = pt_round((127 * xt / xt.abs().max(dim=1, keepdim=True)[0])).to(torch.int8)#.floor()
    #x_real = 127 * xt / xt.abs().max(dim=1, keepdim=True)[0]
    #x_real = (127 * xt / xt.abs().max(dim=1, keepdim=True)[0])
    maxs = xt.abs().max(dim=1)[0]

    print((maxs == out[1]).float().mean())
    print((x_real.t() == out[0]).float().mean())
    # print((x_real[0, :] == out[0][0, :]).float().mean())

    # print((x_real[0, :] != out[0][0, :]).nonzero())
    # print(x_real[0, 17], out[0][0, 17])
    # print(x_real[0, 69], out[0][0, 69])

    print(f"time: {(end - start) / repeat * 1000:.3f} ms")
    exit()


    print(torch.allclose(out[0][0], x_real[0]))
    print( (out[0] ==  x_real).float().mean() )
    print(torch.allclose(out[1], maxs))
    #print(out[1][0], maxs[0])
    print(out[0][0], x_real[0])

    import pdb; pdb.set_trace()

    print(f"time: {(end - start) / repeat * 1000:.3f} ms")
