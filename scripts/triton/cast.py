import torch
import time
import triton
import triton.language as tl
from triton.ops.matmul_perf_model import early_config_prune, estimate_matmul_time


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()


def get_configs_io_bound():
    configs = []
    for num_stages in [2, 3, 4, 5, 6]:
        for block_m in [16, 32]:
            for block_k in [32, 64]:
                for block_n in [32, 64, 128, 256]:
                    num_warps = 2 if block_n <= 64 else 4
                    configs.append(
                        triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': 1},
                                      num_stages=num_stages, num_warps=num_warps))
                    # split_k
                    for split_k in [2, 4, 8, 16]:
                        configs.append(triton.Config({'BLOCK_M': block_m, 'BLOCK_N': block_n, 'BLOCK_K': block_k, 'SPLIT_K': split_k},
                                                     num_stages=num_stages, num_warps=num_warps, pre_hook=init_to_zero('C')))
    return configs


@triton.autotune(
    configs=[
        # basic configs for compute-bound matmuls
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
        # good for int8
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64, 'SPLIT_K': 1}, num_stages=5, num_warps=2),
    ] + get_configs_io_bound(),
    key=['M', 'N', 'K'],
    prune_configs_by={
        'early_config_prune': early_config_prune,
        'perf_model': estimate_matmul_time,
        'top_k': 10
    },
)
@triton.heuristics({
    'EVEN_K': lambda args: args['K'] % (args['BLOCK_K'] * args['SPLIT_K']) == 0,
})
@triton.jit
def _kernel(A, B, C, maxptr1, maxptr2, M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
            GROUP_M: tl.constexpr, SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
            ACC_TYPE: tl.constexpr
            ):
    # matrix multiplication
    pid = tl.program_id(0)
    pid_z = tl.program_id(1)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    # re-order program ID for better L2 performance
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = min(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // (group_size)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    ram = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    rbn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)
    rk = pid_z * BLOCK_K + tl.arange(0, BLOCK_K)
    # pointers
    A = A + (ram[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rbn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K * SPLIT_K)):
        if EVEN_K:
            a = tl.load(A)
            b = tl.load(B)
        else:
            k_remaining = K - k * (BLOCK_K * SPLIT_K)
            a = tl.load(A, mask=rk[None, :] < k_remaining, other=0.)
            b = tl.load(B, mask=rk[:, None] < k_remaining, other=0.)
        acc += tl.dot(a, b)
        A += BLOCK_K * SPLIT_K * stride_ak
        B += BLOCK_K * SPLIT_K * stride_bk

    #print(x_factor.shape, acc.shape)

    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm < M)[:, None] & (rn < N)[None, :]

    w_factor = tl.load(maxptr2)
    x_factor = tl.load(maxptr1 + rn)[None, :]
    
    #acc = w_factor * x_factor * acc.to(C.dtype.element_ty) / (127 * 127)
    acc = (w_factor * x_factor * acc.to(tl.float32) / (127 * 127)).to(tl.float16)

    # handles write-back with reduction-splitting
    if SPLIT_K == 1:
        tl.store(C, acc, mask=mask)
    else:
        tl.atomic_add(C, acc, mask=mask)


class _matmul(torch.autograd.Function):
    kernel = _kernel

    _locks = {}

    @staticmethod
    def _call(a, b, max1, max2):
        device = a.device
        # handle non-contiguous inputs if necessary
        if a.stride(0) > 1 and a.stride(1) > 1:
            a = a.contiguous()
        if b.stride(0) > 1 and b.stride(1) > 1:
            b = b.contiguous()
        # checks constraints
        assert a.shape[1] == b.shape[0], "incompatible dimensions"
        M, K = a.shape
        _, N = b.shape
        # allocates output
        # c = torch.empty((M, N), device=device, dtype=torch.int32)
        c = torch.empty((M, N), device=device, dtype=torch.float16)
        # accumulator types
        ACC_TYPE = tl.int32

        # launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), META['SPLIT_K'])
        _kernel[grid](a, b, c, max1, max2, M, N, K,
                      a.stride(0), a.stride(1),
                      b.stride(0), b.stride(1),
                      c.stride(0), c.stride(1),
                      GROUP_M=8, ACC_TYPE=ACC_TYPE)
        return c

    @staticmethod
    def forward(ctx, a, b, max1, max2):
        return _matmul._call(a, b, max1, max2)


int32_acc_matmul = _matmul.apply

@triton.jit
def _cast_and_quantize(
    x_ptr,
    output_ptr,
    output_maxs,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    abs_x = tl.abs(x)
    max_val = tl.max(input=abs_x, axis=0)
    out = 127 * x / max_val
    output = out.to(tl.int8)
    tl.store(output_ptr + offsets, output, mask=mask)
    tl.store(output_maxs + pid, max_val)

def cast_and_quantize(x: torch.Tensor):
    output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
    output_maxs = torch.empty(x.shape[0], device='cuda', dtype=torch.float16)

    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _cast_and_quantize[grid](x, output, output_maxs, n_elements, BLOCK_SIZE=1024)
    return output, output_maxs


@triton.jit
def _decast_and_dequantize(
    x_ptr,
    maxs,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    max = tl.load(maxs + pid)
    out = x / 127
    output = out.to(tl.float16)
    tl.store(output_ptr + offsets, output, mask=mask)

def decast_and_dequantize(x: torch.Tensor, maxs: torch.Tensor):
    output = torch.empty(*x.shape, device='cuda', dtype=torch.float16)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _decast_and_dequantize[grid](x, maxs, output, n_elements, BLOCK_SIZE=1024)
    return output


@triton.jit
def _cast_global(
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

def cast_global(x: torch.Tensor):
    absmax = x.abs().max().unsqueeze(0)
    output = torch.empty(*x.shape, device='cuda', dtype=torch.int8)
    assert x.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _cast_global[grid](x, absmax, output, n_elements,  BLOCK_SIZE=1024*4)
    return output.t(), absmax
    #return (127 * x / absmax).to(torch.int8), absmax

# @triton.jit
# def _cast_global_t(
#     x_ptr,
#     absmax_ptr,
#     output_ptr,
#     n_elements,
#     N: tl.constexpr,
#     M: tl.constexpr,
#     BLOCK_SIZE: tl.constexpr,
# ):
#     row = tl.program_id(0)
#     col = tl.program_id(1)
#     in_bounds = (row < M) & (col < N)
#     x_val = tl.load(x_ptr[row, col], mask=in_bounds)
#     tl.store(output_ptr[col, row], x_val, mask=in_bounds)

# def cast_global_t(x):
#     absmax = x.abs().max().unsqueeze(0)
#     M, N = x.shape
#     y = torch.empty(N, M, dtype=torch.int8).cuda()
#     _cast_global_t[N, M](x, absmax, y, M, N)
#     return y


def test_all():
    torch.manual_seed(0)
    size = (256*1, 1024)
    x = 0.05 * torch.randn(*size, device='cuda', dtype=torch.float16)

    print(x.shape)
    import time
    repeat = 16

    w = 0.01 * torch.randn(1024*4, 1024, device='cuda', dtype=torch.float16)

    output_triton = x.to(torch.int8)
    w_out = w.to(torch.int8)

def test_all():
    ##############################################
    # torch.manual_seed(0)
    # M = 256*256
    # N = 1024
    # K = 1024*4
    # AT = False
    # BT = True
    # a = 0.01 * torch.randn((K, M) if AT else (M, K), device="cuda", dtype=torch.float16)
    # b = 0.05 * torch.randn((N, K) if BT else (K, N), device="cuda", dtype=torch.float16)
    # a = a.t() if AT else a
    # b = b.t() if BT else b
    # absmax_a = a.abs().max()
    # absmax_b = b.abs().max()
    # a_int8 = (127 * a / absmax_a).floor().to(torch.int8)
    # b_int8 = (127 * b / absmax_b).floor().to(torch.int8)
    # th_c = torch.matmul(a, b)
    # repeat = 16
    # x = a 
    # w = b
    # output_triton = a_int8
    # w_out = b_int8
    ##############################################
    # torch.manual_seed(0)
    repeat = 16
    size = (256*64, 1024*4)
    x = 0.01 * torch.randn(*size, device='cuda', dtype=torch.float16)
    #w = torch.randn(1024,1024*4, device='cuda', dtype=torch.float16).t()
    w = 0.01 * torch.randn(1024, 1024*4, device='cuda', dtype=torch.float16)
    wt = w.t()
    #w = 0.05 * torch.randn(1024*4, 1024, device='cuda', dtype=torch.float16)
    #w_out = w.t().to(torch.int8)
    ##############################################
    # print(output_triton.shape)
    # print(w.shape)
    # # torch.Size([65536, 4096])
    # # torch.Size([4096, 1024])


    ################## -1. triton cast ###################
    for _ in range(8):
        w_out, w_max = cast_global(w)

    triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_matmul_int8_graph):
        w_out, w_max = cast_global(w)

    triton_matmul_int8_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_matmul_int8_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    print(f"w cast: {(end - start) / repeat * 1000:.3f} ms")
    w_cast_time = (end - start) / repeat

    ################### 0. triton cast ###################
    for _ in range(8):
        output_triton, maxs = cast_and_quantize(x)

    triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_matmul_int8_graph):
        output_triton, maxs = cast_and_quantize(x)

    triton_matmul_int8_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_matmul_int8_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    print(f"x cast: {(end - start) / repeat * 1000:.3f} ms")
    x_cast_time = (end - start) / repeat

    # ################### 0. triton uncast ###################
    for _ in range(8):
        c_out = int32_acc_matmul(output_triton, w_out, maxs, w_max)

    triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(triton_matmul_int8_graph):
        c_out = int32_acc_matmul(output_triton, w_out, maxs, w_max)

    triton_matmul_int8_graph.replay()

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        triton_matmul_int8_graph.replay()
    torch.cuda.synchronize()
    end = time.time()

    print(f"multiply: {(end - start) / repeat * 1000:.3f} ms")
    multiply_time = (end - start) / repeat


    # ################### 0. triton uncast ###################
    # for _ in range(8):
    #     new_out = decast_and_dequantize(output_triton, maxs)

    # triton_matmul_int8_graph = torch.cuda.CUDAGraph()
    # with torch.cuda.graph(triton_matmul_int8_graph):
    #     new_out = decast_and_dequantize(output_triton, maxs)

    # triton_matmul_int8_graph.replay()

    # torch.cuda.synchronize()
    # start = time.time()
    # for _ in range(repeat):
    #     triton_matmul_int8_graph.replay()
    # torch.cuda.synchronize()
    # end = time.time()

    # print(f"triton int8 uncast: {(end - start) / repeat * 1000:.3f} ms")



    ################### 2. torch matmul fp16 ###################
    for _ in range(8):
        th_c = torch.matmul(x, wt)

    torch_matmul_fp16_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(torch_matmul_fp16_graph):
        th_c = torch.matmul(x, wt)

    torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        torch_matmul_fp16_graph.replay()
    torch.cuda.synchronize()
    end = time.time()
    print(f"torch fp16 matmul: {(end - start) / repeat * 1000:.3f} ms")
    fp16_time = (end - start) / repeat

    # print(c_out)
    # print(th_c)

    int8_time = x_cast_time + w_cast_time + multiply_time
    print('overall the time for int8', int8_time)
    print('fp16 time', fp16_time)

    print('speedup', -100 * (int8_time - fp16_time) / fp16_time)
test_all()