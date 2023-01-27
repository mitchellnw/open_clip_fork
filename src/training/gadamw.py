import numpy as np
import torch

# GAdamW
'''
    Copyright (c) Meta Platforms, Inc. and its affiliates.
    This source code is licensed under the MIT license found in the
    LICENSE file in the root directory of this source tree.
    
    PyTorch implementation of GAdamW optimizer described in [1].
    [1] Sho Yaida and Susan Zhang,
        "Group-wise AdamW," 2023.
        [arxiv:2301.xxxxx](https://arxiv.org/abs/2301.xxxxx)
'''

class GAdamW(torch.optim.Optimizer):
    '''
    Args:
        lr: global learning rate. Default: 0.001 but please tune
        weight_decay: weight decay. Default: 0.01 but please tune
        beta1: "momentum" for running gradient average. Default: 0.9
        beta2: "momentum" for running group-gradient-square average. Default: 0.999
        eps: regularizing the denominator. Default: 1e-8
    '''

    # Setting things up
    def __init__(self, params, lr=0.001, weight_decay=0.01, betas=(0.9, 0.999), eps=1e-8):
        #params =({'params': p} for p in params)
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(GAdamW, self).__init__(params, defaults)
        self.eps=eps
        self.it_count=0.0
        self.initial_step=True # a flag for an initial step

        print('Using GAdamW')
        
        
    def __setstate__(self, state):
        super(GAdamW, self).__setstate__(state)
        
    # One GAdamW step
    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()
            
        # At the initial step, count the number of groups (of model parameters) while printing out group structure for each group
        if self.initial_step:
            print('Model parameters are partitioned into groups, with tensor structures given by:')
            nG=0
            for group in self.param_groups:
                for p in group['params']:
                    print(p.size())
                    nG=nG+1.0
            print('And the number of groups (of model parameters) is', nG)
            self.initial_step=False # switching off the flag
        
        self.it_count=self.it_count+1.0
        
        for group in self.param_groups:

            # GAdamW hyperparameters for a group
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1 = group['beta1']
            beta2 = group['beta2']
            
            # GAdamW step
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                if p.device.type == "cpu":
                    theta = p.data.cuda(non_blocking=True)
                else:
                    theta=p.data
                    
                param_state = self.state[p]
                g = p.grad.data
                gg = torch.norm(g)**2

                # create velocity (v) and group-gradient-square (li) degrees of freedom
                if 'velocity' not in param_state:
                    v = param_state['velocity'] = torch.zeros_like(theta)
                    li = param_state['lambda_inverse'] = 0.0

                # or read the velocity (v) and group-gradient-square (li) from memory
                else:
                    v = param_state['velocity']
                    li = param_state['lambda_inverse']
                    
                # update the velocity (v) and group-gradient-square (li)
                v = v.mul_(beta1).add_(g,alpha=1.0-beta1)
                li = li*beta2+(1.0-beta2)*(gg.item()/p.nelement())

                # now update the position
                bias_correction1=1.0-beta1**self.it_count
                bias_correction2=1.0-beta2**self.it_count
                bias_correction2_sqrt=np.sqrt(bias_correction2)
                denom = (np.sqrt(li)/bias_correction2_sqrt)+self.eps
                theta = theta.mul_(1.0-lr*weight_decay).add_(v,alpha=-lr/(denom*bias_correction1))

                # save current params
                param_state['velocity'] = v
                param_state['lambda_inverse'] = li
