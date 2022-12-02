import numpy as np


def assign_learning_rate(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        assign_learning_rate(optimizer, lr)
        return lr
    return _lr_adjuster


def linearly_warmup_beta2(optimizer, init_beta2, final_beta2, warmup_length, steps):
    def _beta2_adjuster(step):
        if step < warmup_length:
            mixing_coef = (step / float(warmup_length))
            beta2 = (1 - mixing_coef) * init_beta2 + mixing_coef * final_beta2
        else:
            beta2 = final_beta2
        for param_group in optimizer.param_groups:
            param_group["betas"] = (param_group["betas"][0], beta2)
        return beta2
    return _beta2_adjuster

def get_batch_size(step, batch_size, warmup_length):
    return int(
        max(2, 
            batch_size * min(1, (step + 1) / warmup_length)
        )
    )