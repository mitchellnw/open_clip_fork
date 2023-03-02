
import torch
import bitsandbytes as bnb
import numpy as np

def replace_linear(model, linear_replacement, skip_modules=["lm_head", "conv1", "embedding"], copy_weights=True):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        linear_replacement (`torch.nn.Module`):
            The linear module that replaces the old one. Only expects standard arguments.
            If other arguments need to be passed, use a lambda.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
        copy_weights (`bool`):
            Copy the weights from the old linear module to the new one
    """
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_linear(module, linear_replacement, skip_modules, copy_weights)

        if isinstance(module, torch.nn.Linear) and name not in skip_modules:
            if name in ['in_proj_linear', 'out_proj', 'c_fc', 'c_proj']:
                old_module = model._modules[name]
                model._modules[name] = linear_replacement(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                )
                if copy_weights:
                    model._modules[name].weight.data.copy_(old_module.weight.data)
                    if model._modules[name].bias is not None:
                        model._modules[name].bias.data.copy_(old_module.bias)

    return model


if __name__ == '__main__':

    err_percent = 0
    lim = 1000

    dim = 750
    loss_scaler = 1.#2**16
    for k in range(lim):

        #x =  0.2 * torch.randn(256, dim).cuda()
        #x = (torch.rand(256, dim) * 2 - 1).cuda()
        x = torch.randn(256, dim).cuda()
        
        fp32model = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
            torch.nn.GELU(),
            torch.nn.Linear(dim, dim),
        ).cuda()

        y1 = fp32model(x) + x
        y1.mean().backward()

        int8realmodel = torch.nn.Sequential(
            bnb.nn.Linear8bitLt(dim, dim),
            torch.nn.GELU(),
            bnb.nn.Linear8bitLt(dim, dim),
            torch.nn.GELU(),
            bnb.nn.Linear8bitLt(dim, dim),
        ).cuda()
        int8realmodel[0].weight.data.copy_(fp32model[0].weight.data)
        int8realmodel[2].weight.data.copy_(fp32model[2].weight.data)
        int8realmodel[4].weight.data.copy_(fp32model[4].weight.data)
        int8realmodel[0].bias.data.copy_(fp32model[0].bias.data)
        int8realmodel[2].bias.data.copy_(fp32model[2].bias.data)
        int8realmodel[4].bias.data.copy_(fp32model[4].bias.data)

        with torch.cuda.amp.autocast():
            y2 = int8realmodel(x) + x

        (loss_scaler * y2).mean().backward()


        int8simmodel = torch.nn.Sequential(
            bnb.nn.LinearInt8(dim, dim),
            torch.nn.GELU(),
            bnb.nn.LinearInt8(dim, dim),
            torch.nn.GELU(),
            bnb.nn.LinearInt8(dim, dim),
        ).cuda()
        int8simmodel[0].weight.data.copy_(fp32model[0].weight.data)
        int8simmodel[2].weight.data.copy_(fp32model[2].weight.data)
        int8simmodel[4].weight.data.copy_(fp32model[4].weight.data)
        int8simmodel[0].bias.data.copy_(fp32model[0].bias.data)
        int8simmodel[2].bias.data.copy_(fp32model[2].bias.data)
        int8simmodel[4].bias.data.copy_(fp32model[4].bias.data)

        with torch.cuda.amp.autocast():
            y3 = int8simmodel(x) + x
            
        (loss_scaler * y3).mean().backward()

        g1 = fp32model[0].weight.grad
        g2 = int8realmodel[0].weight.grad / loss_scaler
        g3 = int8simmodel[0].weight.grad / loss_scaler

        #print(g1[0, 0:1], g2[0, 0:1], g3[0, 0:1])

        real_error =  (g1 - g2).abs().mean()
        sim_error = (g1 - g3).abs().mean()
        print('error, real:', real_error)
        print('error, sim:', sim_error)

        err_percent += (sim_error < real_error).float().item()
        print('running mean for sim better than real', 100. * err_percent / float(k + 1))

    err_percent /= lim
    print('Sim better than real percent', 100 * err_percent)




"""


        # not supported by PyTorch. TODO: create work-around
        #if req_gradB: grad_B = torch.matmul(grad_output.t(), A)
        if req_gradA: grad_A = torch.matmul(fp8out, B.t().to(fp8out.dtype)).to(fp8A.dtype)
        if req_gradB:
            if fp8A.ndim == 3:
                fp8At = fp8A.transpose(2, 1)
            elif fp8A.ndim == 2:
                fp8At = fp8A.t()
            grad_B = torch.matmul(fp8At.to(fp8out.dtype), fp8out).to(B.dtype)
"""