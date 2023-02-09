import numpy as np
import torch

class StableAdamW(torch.optim.Optimizer):

    # Setting things up
    def __init__(self, params, lr=0.004, weight_decay=0.2, betas=(0.9, 0.999), eps=1e-6):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(StableAdamW, self).__init__(params, defaults)

        self.initial_step=True # a flag for an initial step
        self.eps=eps
        self.it_count=0.0
        for group in self.param_groups:
            group['step'] = 1.
        
        print('Using StableAdamW-v2')


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
                theta=p.data
                param_state = self.state[p]
                g = p.grad.data

                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                    param_state['rms_mean'] = 0.
                    param_state['rms_std'] = 0.
                    param_state['rms_min'] = 0.
                    param_state['rms_max'] = 0.
                    param_state['beta2hat'] = 0.
                    param_state['k'] = 1.
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']
                    k = param_state['k']

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = 1. - (param_state['k'] ** (-0.8))
                    
                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g,g,value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)

                theta = theta.mul_(1.0-lr*weight_decay).addcdiv_(v, denominator, value=-lr)

                # for logging
                rms = torch.div(
                    g.pow(2), 
                    torch.maximum(u, (self.eps ** 2) * torch.ones_like(u))
                )
                param_state['rms_mean'] = rms.mean().item()
                param_state['rms_std'] = rms.std().item()
                param_state['rms_min'] = rms.min().item()
                param_state['rms_max'] = rms.max().item()
                param_state['rms_sq_d1'] = (rms - 1).pow(2).mean().item()
                param_state['rms_d1'] = (rms.sqrt() - 1).pow(2).mean().item()
                param_state['numel'] = rms.numel()
                param_state['relu'] = torch.nn.functional.relu(rms - 1).mean().item()
                param_state['beta2hat'] = beta2hat

                # v0
                # param_state['err'] = max(rms.mean().sqrt().item() - 1, 0)
                # v1
                # param_state['err'] = rms.max().sqrt().item()
                # v2
                param_state['err'] = rms.mean().sqrt().item()

                param_state['k'] = max(1., param_state['k'] + 1 - beta2 * param_state['err'])

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u
                #param_state['lambda_inverse'] = li

                # if self.rank == 0:
                #     import pdb; pdb.set_trace()
            
            group['step'] = step + 1

                
