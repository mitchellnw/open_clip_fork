import numpy as np
import torch

class G1AdamW(torch.optim.Optimizer):

    # Setting things up
    def __init__(self, params, lr=0.004, weight_decay=0.2, betas=(0.9, 0.999), eps=1e-6):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(G1AdamW, self).__init__(params, defaults)

        self.initial_step=True # a flag for an initial step
        self.eps=eps
        self.it_count=0.0
        self.d = 1. # clip thresh.
        for group in self.param_groups:
            group['step'] = 1.
        
        print('Using G1AdamW-v1')


    def __setstate__(self, state):
        super(G1AdamW, self).__setstate__(state)
        
    # One G1AdamW step
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

                if self.precision == 'custom_fp16':
                    g = p.grad.data / self.custom_scaler
                    if torch.any(torch.isnan(g) | torch.isinf(g)):
                        continue
                else:
                    g = p.grad.data
                    

                torch.distributed.barrier()

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                    param_state['rms_mean'] = 0.
                    param_state['rms_std'] = 0.
                    param_state['rms_min'] = 0.
                    param_state['rms_max'] = 0.
                    param_state['beta2hat'] = 0.
                    z = param_state['run_g2'] = 0.
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']
                    z = param_state['run_g2']

                

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)

                gnorm = g.pow(2).sum().sqrt()
                z = beta2hat * z + (1 - beta2hat) * gnorm
                param_state['run_g2'] = z
                g = g / (z + self.eps)
                    
                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g,g,value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)

                theta = theta.mul_(1.0-lr*weight_decay).addcdiv_(
                    v, 
                    denominator, 
                    value=-lr
                )

                param_state['beta2hat'] = beta2hat

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u
                #param_state['lambda_inverse'] = li
            
            group['step'] = step + 1

                
