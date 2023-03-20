import torch
import torch.nn as nn
from int8_matmul_mixed_dequanitze import int8_matmul_mixed_dequanitze
from quantize_global import quantize_global
from quantize_rowwise_nogroup import quantize_rowwise_nogroup

class _switchback(torch.autograd.Function):

    @staticmethod
    def forward(ctx, X, W):
        X_int8, state_x = quantize_rowwise_nogroup(X)
        W_int8, state_w = quantize_global(W)
        return int8_matmul_mixed_dequanitze(X_int8, W_int8, state_x, state_w)
    
    def backward(ctx, G):
        return None, None
    
class SwitchBackLinear(nn.Module):
    def forward(self, x):
        return _switchback.apply(x, self.weight)
    


if __name__ == '__main__':

    fp16_linear = nn.Linear(2048, 2048)
