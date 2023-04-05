import torch
import math
import time
try:
    import triton
    import triton.language as tl
except ImportError as e:
    print('triton is not installed, please install by running `pip install triton -U --pre`')
    exit()

@triton.autotune(configs = [
    triton.Config({}, num_warps = 4),

], key = ['n_elements'])
@triton.jit
def update_fn_kernel(
    p_ptr,
    grad_ptr,
    exp_avg_ptr,
    exp_avg2_ptr,
    lr,
    wd,
    beta1,
    beta2,
    eps,
    update_clip,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    smem_numerator,
    semem_denominator,
    smem_numerator_debug,
    semem_denominator_debug,
    rms_debug,
    num_kernels,
    num_kernels_p2 : tl.constexpr,
):
    pid = tl.program_id(axis = 0)

    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_elements

    # offsetted pointers

    offset_p_ptr = p_ptr + offsets
    offset_grad_ptr = grad_ptr + offsets
    offset_exp_avg_ptr = exp_avg_ptr + offsets
    offset_exp_avg2_ptr = exp_avg2_ptr + offsets

    # load, early exit if nan
    grad = tl.load(offset_grad_ptr, mask = mask)
    if tl.max(tl.libdevice.isnan(grad), 0) >= 1:
        return

    
    
    exp_avg2 = tl.load(offset_exp_avg2_ptr, mask = mask)

    # stepweight decay

    

    # update exp_avgs
    g2 = grad * grad
    exp_avg2 = beta2 * exp_avg2 + (1 - beta2) * g2

    
    if update_clip:
        ratio = tl.where(mask, g2 / tl.maximum(exp_avg2, eps * eps), 0)
        rms_numerator = tl.sum(ratio, axis=0)
        rms_denominator = tl.sum(mask, axis=0)
        tl.store(smem_numerator + pid, rms_numerator)
        tl.store(semem_denominator + pid, rms_denominator)
        smem_arange = tl.arange(0, num_kernels_p2)

        p = tl.load(offset_p_ptr, mask = mask)
        p = p * (1 - lr * wd)
        exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad


        total_rms_numerator = tl.sum(tl.load(smem_numerator + smem_arange), axis=0)
        total_rms_denominator = tl.sum(tl.load(semem_denominator + smem_arange), axis=0)
        rms = tl.sqrt(total_rms_numerator / total_rms_denominator)
        #rms = 1.4292508363723755
        eta = lr / tl.maximum(rms, 1.)

        tl.store(smem_numerator_debug + pid, total_rms_numerator)
        tl.store(semem_denominator_debug + pid, total_rms_denominator)
        tl.store(rms_debug + pid, rms)
    else:
        p = tl.load(offset_p_ptr, mask = mask)
        p = p * (1 - lr * wd)
        exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)
        exp_avg = beta1 * exp_avg + (1 - beta1) * grad
        eta = lr

    p = p - eta * (exp_avg / (tl.sqrt(exp_avg2) + eps))

    # store new params and momentum running average coefficient

    tl.store(offset_p_ptr, p, mask = mask)
    tl.store(offset_exp_avg_ptr, exp_avg, mask = mask)
    tl.store(offset_exp_avg2_ptr, exp_avg2, mask = mask)


def update_fn(
    p: torch.Tensor,
    grad: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg2: torch.Tensor,
    lr: float,
    wd: float,
    beta1: float,
    beta2: float,
    eps: float,
    update_clip : bool = True,
):
    assert all([t.is_cuda for t in (p, grad, exp_avg)])
    n_elements = p.numel()
    BLOCK_SIZE = 4096
    num_kernels = math.ceil(n_elements / BLOCK_SIZE)
    num_kernels_p2 = int(2 ** (math.ceil(math.log2(num_kernels))))
    smem_numerator = torch.zeros(num_kernels_p2, dtype=torch.float32, device=p.device)
    smem_denominator = torch.zeros(num_kernels_p2, dtype=torch.int32, device=p.device)
    smem_numerator_debug = torch.zeros(num_kernels_p2, dtype=torch.float32, device=p.device)
    smem_denominator_debug = torch.zeros(num_kernels_p2, dtype=torch.int32, device=p.device)
    smem_rms_debug = torch.zeros(num_kernels_p2, dtype=torch.float32, device=p.device)

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)    

    update_fn_kernel[grid](
        p,
        grad,
        exp_avg,
        exp_avg2,
        lr,
        wd,
        beta1,
        beta2,
        eps,
        update_clip,
        n_elements,
        BLOCK_SIZE,
        smem_numerator,
        smem_denominator,
        smem_numerator_debug,
        smem_denominator_debug,
        smem_rms_debug,
        num_kernels,
        num_kernels_p2,
    )

    return smem_numerator, smem_denominator, smem_numerator_debug, smem_denominator_debug, smem_rms_debug

def test_update(
        p,
        grad,
        exp_avg,
        exp_avg2,
        lr,
        wd,
        beta1,
        beta2,
        eps,
        n_elements
    ):

    

    beta1hat = beta1 #* (1 - beta1**(step - 1)) / (1 - beta1**step)
    beta2hat = beta2 #* (1 - beta2**(step - 1)) / (1 - beta2**step)

    g = grad
        
    v = exp_avg.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
    u = exp_avg2.mul_(beta2hat).addcmul_(g,g,value=1.0-beta2hat)

    denominator = u.sqrt().add_(eps)
        
    #import pdb; pdb.set_trace()
    # for logging
    rms = torch.div(
        g.pow(2), 
        torch.maximum(u, (eps ** 2) * torch.ones_like(u))
    )
    print('rms', rms.mean().sqrt().item())
    #rms = torch.tensor(1.)

    #p = p.mul_(1.0-lr*wd)
    p = p.mul_(1.0-lr*wd).addcdiv_(
        v, 
        denominator, 
        value=-lr * (1. / max(1., rms.mean().sqrt().item()))
    )

    return p, v, u

import numpy as np
import torch

class TAdamW(torch.optim.Optimizer):

    # Setting things up
    def __init__(self, params, lr=0.004, weight_decay=0.2, betas=(0.9, 0.999), eps=1e-6, update_clip=True):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(TAdamW, self).__init__(params, defaults)

        self.initial_step=True # a flag for an initial step
        self.eps=eps
        self.it_count=0.0
        self.update_clip = update_clip
        for group in self.param_groups:
            group['step'] = 1.
        
        print('Using TAdamW-v1')


    def __setstate__(self, state):
        super(TAdamW, self).__setstate__(state)
        
    # One TAdamW step
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
            

        for group in self.param_groups:


            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            step = group['step']
            

            for p in group['params']:
                if p.grad is None:
                    continue
                theta=p.data
                param_state = self.state[p]
                g = p.grad.data
                    

                torch.distributed.barrier()

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)
                
                update_fn(p, g, v, u, lr, weight_decay, beta1hat, beta2hat, self.eps, self.update_clip)
            
            group['step'] = step + 1

                



if __name__ == '__main__':
    test_u = torch.randn(1000, 1025).cuda().abs()
    test_v = torch.randn(1000, 1025).cuda()
    test_p = torch.randn(1000, 1025).cuda()
    test_g = torch.randn(1000, 1025).cuda()

    test_u2 = test_u.clone()
    test_v2 = test_v.clone()
    test_p2 = test_p.clone()
    test_g2 = test_g.clone()

    oldp = test_p.clone()
    oldu = test_u.clone()
    oldv = test_v.clone()

    beta1 = 0.9
    beta2 = 0.95
    eps = 1e-6
    lr = 1e-1
    wd = 0.1
    numel = test_p.numel()

    p1, v1, u1 = test_update(
        test_p, test_g, test_v, test_u, lr, wd, beta1, beta2, eps, numel
    )

    out1, out2, out3, out4, out5 = update_fn(
        test_p2, test_g2, test_v2, test_u2, lr, wd, beta1, beta2, eps, True,
    )

    print('est rms', (out1.sum() / out2.sum()).sqrt().item())

    #print((oldp - test_p2).abs().mean())
    print((p1 - test_p2).abs().mean())
    print((test_v2 - v1).abs().mean())
    print((test_u2 - u1).abs().mean())

    # if (p1 - test_p2).abs().mean() < 1e-7:
    #     print('passed')
    # else:
    import pdb; pdb.set_trace()


    repeat = 32
    for _ in range(repeat // 2):
        update_fn(
            test_p2, test_g2, test_v2, test_u2, lr, wd, beta1, beta2, eps, True,
        )

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        update_fn(
            test_p2, test_g2, test_v2, test_u2, lr, wd, beta1, beta2, eps, True,
        )

    torch.cuda.synchronize()
    end = time.time()
    ms = (end - start) / repeat * 1000
    print(f"time: {ms:.3f} ms")

