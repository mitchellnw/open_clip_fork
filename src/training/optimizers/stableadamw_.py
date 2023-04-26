import numpy as np
import torch

class StableAdamW(torch.optim.Optimizer):

    def __init__(self, params, lr=0.004, weight_decay=0.2, betas=(0.9, 0.999), eps=1e-6):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(StableAdamW, self).__init__(params, defaults)

        self.initial_step=True
        self.eps=eps
        self.it_count=0.0
        self.d = 1. # clip thresh -- if you still get issues, you can decrease.
        for group in self.param_groups:
            group['step'] = 1.
        
        print('Using StableAdamW-v1')


    def __setstate__(self, state):
        super(StableAdamW, self).__setstate__(state)
        
    # One StableAdamW step
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
                if torch.any(torch.isnan(g) | torch.isinf(g)):
                    continue
                theta=p.data
                param_state = self.state[p]
                g = p.grad.data
                    
                #torch.distributed.barrier()

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)
                    
                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g,g,value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)
                    
                rms = torch.div(
                    g.pow(2), 
                    torch.maximum(u, (self.eps ** 2) * torch.ones_like(u))
                ).mean().sqrt().item()

                theta = theta.mul_(1.0-lr*weight_decay).addcdiv_(
                    v, 
                    denominator, 
                    value=-lr * (1. / max(1., rms / self.d ))
                )

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u

            
            group['step'] = step + 1
