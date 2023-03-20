# gelu implemented in triton

import triton
import torch

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608028654 * (x + 0.044715 * torch.pow(x, 3))))
