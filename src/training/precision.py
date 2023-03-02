import torch
from contextlib import suppress


def get_autocast(precision):
    if precision == 'amp' or precision == 'custom_fp16':
        print('amp fp16')
        return torch.cuda.amp.autocast
    elif precision == 'amp_bfloat16' or precision == 'amp_bf16':
        # amp_bfloat16 is more stable than amp float16 for clip training
        print('amp bfloat16')
        return lambda: torch.cuda.amp.autocast(dtype=torch.bfloat16)
    else:
        print('ERROR!')
        return suppress
