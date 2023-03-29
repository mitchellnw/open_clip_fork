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
    x = tl.load(x_ptr + tl.arange(0, 4))
    y = tl.load(x_ptr + tl.arange(0, 4) * N)
    out = x 
    #tl.store(output_ptr + tl.arange(0, 4)[None, :], out[None, :] )
    #tl.store(output_ptr + N * tl.arange(0, 4)[:, None], out[:, None] )
    tl.store(output_ptr + tl.arange(0, 4)[None, :] + N * tl.arange(0, 4)[:, None], x[None, :] * y[:, None])

    # row_mask = arange < BLOCK_SIZE
    # abs_x = tl.abs(x)
    # max_val = tl.max(tl.where(row_mask, abs_x, -1), axis=0)
    # output = tl.libdevice.llrint(127. * (x / max_val))
    # new_offsets =  pid + tl.arange(0, P2) * M
    # tl.store(output_ptr + new_offsets, output, mask=new_offsets < N * M)
    # tl.store(output_maxs + pid, max_val)

def quantize_rowwise_nogroup_transpose(x: torch.Tensor):
    M, N = x.shape
    output = torch.zeros(M, N, device='cuda', dtype=torch.float16)
    output_maxs = torch.empty(M, device='cuda', dtype=torch.float16)
    P2 = int(2 ** (math.ceil(math.log2(N))))

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (1,)#(triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
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

    x = torch.randn(1024, 1024).cuda().to(torch.float16)
    out = quantize_rowwise_nogroup_transpose(x)

    print(out[0][:4, :4])
    import pdb; pdb.set_trace()

