import numpy as np
import torch

# SAdamW
'''
    Copyright (c) Meta Platforms, Inc. and its affiliates.
    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.

    AdamW, but recording some things
'''

class SAdamW(torch.optim.Optimizer):
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
        super(SAdamW, self).__init__(params, defaults)

        self.initial_step=True # a flag for an initial step
        self.eps=eps
        self.it_count=0.0
        for group in self.param_groups:
            group['step'] = 1.

        self.power = -np.log(1. - 0.999) / np.log(8000)
        

        print('Using SAdamW')
    def __setstate__(self, state):
        super(SAdamW, self).__setstate__(state)
    
    # Jot down the loss value
    def jot(self, loss_value):
        self.loss_value=loss_value
        
    # One SAdamW step
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
            #for k in group: print(k)
            #running_k = group['running_k']
            

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
                    running_k = param_state['running_k'] = 1.
                    # li = gg.item()/p.nelement()
                    # print(weight_decay,p.size(),li)
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']
                    running_k = param_state['running_k']
                    #li = param_state['lambda_inverse']
                    #li = li*0.9+(0.1)*(gg.item()/p.nelement())

                



                # lets check current g^2 over past u
                # if step == 1:
                #     err = 0
                # else:
                #     g2 = g.pow(2)
                #     old_beta2 = 1. - (running_k ** (-self.power))
                #     old_uhat = u / (1.0 - old_beta2**(step - 1))
                #     ratio = torch.div(g2, old_uhat + (self.eps ** 2))
                #     err = max(ratio.mean().sqrt().item() - 1, 0)
                # param_state['err'] = err
                # param_state['serr'] = beta2*err
                # param_state['running_k'] = max(1, running_k + 1 - beta2*err)


                # if p.grad.numel() == 1 and self.rank == 0:
                #     import pdb; pdb.set_trace()

                # TODO: why does it currently work for scalers..?
                if p.grad.numel() == 1:
                    new_beta2 = 0.9
                else:
                    new_beta2 = 1. - (param_state['running_k'] ** (- self.power))
                new_beta2 = max(new_beta2, 0.5)

                param_state['det_beta2'] = new_beta2
                

                # if self.rank == 0 and step == 2:
                #     import pdb; pdb.set_trace()


                v = v.mul_(beta1).add_(g, alpha=1.0-beta1)
                u = u.mul_(new_beta2).addcmul_(g,g,value=1.0-new_beta2)

                # now update the position
                bias_correction1=1.0#-beta1**step
                bias_correction2=1.0#-new_beta2**step

                vhat = v / bias_correction1
                uhat = u / bias_correction2

                # if self.rank == 0 and step == 10:
                #     import pdb; pdb.set_trace()

                denom = uhat.sqrt().add_(self.eps)

                g2 = g.pow(2)
                #rms = torch.div(g2, uhat + (self.eps ** 2))
                rms = torch.div(g2, uhat + self.eps)
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
                param_state['relu'] = torch.nn.functional.relu(rms - 1).mean().item()
                #param_state['rms_diff0'] = (rms - 1)

                newlr = lr

                #print(rms.mean())
                err = torch.nn.functional.relu(rms - 1).mean().item()
                param_state['err'] = err
                param_state['serr'] = beta2*err
                param_state['running_k'] = running_k + 1 - beta2*err
                #print(param_state['running_k'], group['step'], err, new_beta2)

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
