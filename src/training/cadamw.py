import numpy as np
import torch

# CAdamW
'''
    Copyright (c) Meta Platforms, Inc. and its affiliates.
    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.

    AdamW, but recording some things
'''

class CAdamW(torch.optim.Optimizer):
    '''
    Args:
        lr: global learning rate. Default: 0.004 but please tune
        weight_decay: weight decay. Default: 0.1 but please tune
        beta1: "momentum" for running gradient average. Default: 0.9
        beta2: "momentum" for running group-gradient-square average. Default: 0.999
        eps: regularizing the denominator. Default: 1e-8
    '''

    # Setting things up
    def __init__(self, params, lr=0.004, weight_decay=0.2, betas=(0.9, 0.999), eps=1e-6):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(CAdamW, self).__init__(params, defaults)

        self.initial_step=True # a flag for an initial step
        self.eps=eps
        self.it_count=0.0
        for group in self.param_groups:
            group['step'] = 1.
        
        print('Using CAdamW')
    def __setstate__(self, state):
        super(CAdamW, self).__setstate__(state)
    
    # Jot down the loss value
    def jot(self, loss_value):
        self.loss_value=loss_value
        
    # One CAdamW step
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
            
        #self.it_count=self.it_count+1.0
        
        # At the initial step, count the number of groups (of model parameters) while calculating fan_out & fan_in for each group
        if self.initial_step:
            nG=0.0
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    nG=nG+1.0
                    
            print('the number of groups (of model parameters) is', nG)
            self.initial_step=False # switching off the flag
        
        for group in self.param_groups:

            # ANGEL hyperparameters for a group
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            step = group['step']
            
            # ANGEL step
            for p in group['params']:
                if p.grad is None:
                    continue
                theta=p.data
                param_state = self.state[p]
                g = p.grad.data
                # gg = torch.norm(g)**2


                if 'exp_avg' not in param_state:
                    v = param_state['exp_avg'] = torch.zeros_like(theta)
                    u = param_state['exp_avg_sq'] = torch.zeros_like(theta)
                    param_state['rms_mean'] = 0.
                    param_state['rms_std'] = 0.
                    param_state['rms_min'] = 0.
                    param_state['rms_max'] = 0.
                    param_state['g2_mean'] = 0.
                    param_state['g2_std'] = 0.
                    param_state['g2_min'] = 0.
                    param_state['g2_max'] = 0.
                    # li = gg.item()/p.nelement()
                    # print(weight_decay,p.size(),li)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']
                    #li = param_state['lambda_inverse']
                    #li = li*0.9+(0.1)*(gg.item()/p.nelement())

                
                    
                v = v.mul_(beta1).add_(g, alpha=1.0-beta1)
                u = u.mul_(beta2).addcmul_(g,g,value=1.0-beta2)

                # now update the position
                bias_correction1=1.0-beta1**step
                bias_correction2=1.0-beta2**step

                vhat = v / bias_correction1
                uhat = u / bias_correction2

                # if self.rank == 0 and step == 10:
                #     import pdb; pdb.set_trace()

                denom = uhat.sqrt().add_(self.eps)

                g2 = g.pow(2)
                rms = torch.div(g2, uhat + (self.eps ** 2))
                param_state['g2_mean'] = g2.mean().item()
                param_state['g2_std'] = g2.std().item()
                param_state['g2_min'] = g2.min().item()
                param_state['g2_max'] = g2.max().item()
                param_state['rms_mean'] = rms.mean().item()
                param_state['rms_std'] = rms.std().item()
                param_state['rms_min'] = rms.min().item()
                param_state['rms_max'] = rms.max().item()

                param_state['rms_sq_d1'] = (rms - 1).pow(2).mean().item()
                param_state['rms_d1'] = (rms.sqrt() - 1).pow(2).mean().item()
                param_state['numel'] = rms.numel()
                #param_state['rms_diff0'] = (rms - 1)

                newlr = lr
                if self.rms_scale:
                    real_rms = rms.mean().sqrt().item()
                    newlr = lr / max(1, real_rms)
                    if self.rank == 0:
                        print('here and doing rms scale with', newlr, max(1, real_rms))



                theta = theta.mul_(1.0-lr*weight_decay).addcdiv_(vhat, denom, value=-newlr)

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u
                #param_state['lambda_inverse'] = li
            
            group['step'] = step + 1
            #print('step is', group['step'])
                
    # # Print out things
    # def printout(self):
    #     record_ANGEL_stats=list()
    #     for group in self.param_groups:
    #         for p in group['params']:
    #             param_state = self.state[p]
    #             print(p.size(), param_state['lambda_inverse'])
    #             record_ANGEL_stats.append(param_state['lambda_inverse'])
    #     print(record_ANGEL_stats)
    #     return record_ANGEL_stats
