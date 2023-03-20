import numpy as np
import torch

class MonitorAdamW(torch.optim.Optimizer):

    # Setting things up
    def __init__(self, params, lr=0.004, weight_decay=0.2, betas=(0.9, 0.999), eps=1e-6):
        beta1, beta2 = betas[0], betas[1]
        defaults = dict(lr=lr, weight_decay=weight_decay, beta1=beta1, beta2=beta2)
        super(MonitorAdamW, self).__init__(params, defaults)

        self.initial_step=True # a flag for an initial step
        self.eps=eps
        self.it_count=0.0
        self.d = 1. # clip thresh.
        for group in self.param_groups:
            group['step'] = 1.
        
        print('Using MonitorAdamW-v1')


    def __setstate__(self, state):
        super(MonitorAdamW, self).__setstate__(state)
        
    # One MonitorAdamW step

    # 65504
    # 0.00006103515625
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
                    param_state['underflow'] = 0.
                    param_state['overflow'] = 0.
                else:
                    v = param_state['exp_avg']
                    u = param_state['exp_avg_sq']

                # figure out how bad.
                gabs = g.abs() * 65504
                too_high = (gabs > 65504).float().mean()
                too_low = ((gabs < 0.00006103515625) * (gabs > 0.)).float().mean()
                too_low_v2 = ((gabs < 1e-24) * (gabs > 0.)).float().mean()

                param_state['underflow_1'] = too_low.item()
                param_state['underflow2_1'] = too_low_v2.item()
                param_state['overflow_1'] = too_high.item()

                too_high = (gabs > (65504 / 2)).float().mean()
                too_low = ((gabs < (0.00006103515625 / 2)) * (gabs > 0.)).float().mean()
                too_low_v2 = ((gabs < (1e-24 / 2)) * (gabs > 0.)).float().mean()

                param_state['underflow_2'] = too_low.item()
                param_state['underflow2_2'] = too_low_v2.item()
                param_state['overflow_2'] = too_high.item()

                too_high = (gabs > (65504 / 4)).float().mean()
                too_low = ((gabs < (0.00006103515625 / 4)) * (gabs > 0.)).float().mean()
                too_low_v2 = ((gabs < (1e-24 / 4)) * (gabs > 0.)).float().mean()

                param_state['underflow_4'] = too_low.item()
                param_state['underflow2_4'] = too_low_v2.item()
                param_state['overflow_4'] = too_high.item()

                too_high = (gabs > (65504 / 8)).float().mean()
                too_low = ((gabs < (0.00006103515625 / 8)) * (gabs > 0.)).float().mean()
                too_low_v2 = ((gabs < (1e-24 / 8)) * (gabs > 0.)).float().mean()

                param_state['underflow_8'] = too_low.item()
                param_state['underflow2_8'] = too_low_v2.item()
                param_state['overflow_8'] = too_high.item()

                too_high = (gabs > (65504 / 16)).float().mean()
                too_low = ((gabs < (0.00006103515625 / 16)) * (gabs > 0.)).float().mean()
                too_low_v2 = ((gabs < (1e-24 / 16)) * (gabs > 0.)).float().mean()

                param_state['underflow_16'] = too_low.item()
                param_state['underflow2_16'] = too_low_v2.item()
                param_state['overflow_16'] = too_high.item()

                beta1hat = beta1 * (1 - beta1**(step - 1)) / (1 - beta1**step)
                beta2hat = beta2 * (1 - beta2**(step - 1)) / (1 - beta2**step)
                    
                v = v.mul_(beta1hat).add_(g, alpha=1.0-beta1hat)
                u = u.mul_(beta2hat).addcmul_(g,g,value=1.0-beta2hat)

                denominator = u.sqrt().add_(self.eps)
                    
                # for logging
                rms = torch.div(
                    g.pow(2), 
                    torch.maximum(u, (self.eps ** 2) * torch.ones_like(u))
                )

                theta = theta.mul_(1.0-lr*weight_decay).addcdiv_(
                    v, 
                    denominator, 
                    value=-lr * (1. / max(1., rms.mean().sqrt().item() / self.d ))
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

                # save current params
                param_state['exp_avg'] = v
                param_state['exp_avg_sq'] = u
                #param_state['lambda_inverse'] = li
            
            group['step'] = step + 1

                
