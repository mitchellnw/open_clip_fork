import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def proc(df, lim):
    df.drop(df[df[0] == 0].index[1:], axis=0, inplace=True)
    df.drop_duplicates(0, keep='last', inplace=True)
    if lim > 0:
        df = df[df[0] < lim]
    return df
if __name__ == '__main__':
    kernel_size = 40
    min_loss = 14
    max_scaler = 1
    log_level = 3

    # NOTE: LOOK AT FEATURE STDDEV!

    file_list = []
    file_list = [
        #('clip-bigG14-pd05-pinit-160k-2e-3-amp_bfloat16-v1', 'p-init (blew-up)', 'C0', -1),
        ('clip-bigG14-pd05-160k-2e-3-amp_bfloat16-v1', 'standard (blew-up)', 'C0', 4000),#3900),
        #('clip-bigG14-pd05-ls0-160k-2e-3-amp_bfloat16-v1', 'layer-scale=0', 'C2', -1),
        ('clip-bigG14-pd05-ls1-pinit-160k-2e-3-0.95-amp_bfloat16-v1', 'p-init + layer-scale=1 + beta2=0.95', 'C1', -1),
    ]
    #file_list = file_list[1:2]
    #file_list = file_list[:2]

    fig, axlist = plt.subplots(log_level, 1, figsize=(8, 5 * log_level))
    if log_level == 1:
        axlist = [axlist]
    for j, (file, name, color, lim) in enumerate(file_list):

        if log_level >= 1:
            ax = axlist[0]
            for i in range(1):
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/open_clip/{file}/data/{i}/loss.csv', names=list(range(2)))
                df = proc(df, lim)
                ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), alpha=0.5, color=color)
                
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name)
                ax.set_ylabel('Loss')

        
        if log_level >= 2:
            ax = axlist[1]
            for i in range(1):
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/open_clip/{file}/data/{i}/features-module.visual.transformer.resblocks.40.csv', names=list(range(4)))
                df = proc(df, lim)
                #df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/open_clip/{file}/data/{i}/features-module.transformer.resblocks.20.csv', names=list(range(4)))#.drop_duplicates(0, keep='last')


                ax.plot(df.iloc[:, 0], df.iloc[:, 2], color=color)
                #ax.plot(df.iloc[:, 0], df.iloc[:, 1], color=color, alpha=0.3)
                ax.plot(df.iloc[:, 0], df.iloc[:, 3], color=color, alpha=0.6)
                ax.set_yscale('log')
                ax.set_ylabel('Feature mean and max (block 40)')

                

        #print(f'/fsx/home-mitchellw/experimetns/open_clip/{file}/data/{i}/params-module.transformer.resblocks.30.attn.out_proj.weight.csv')
        if log_level >= 3:
            ax = axlist[2]
            for i in range(1):
                #layer = 'params-module.logit_scale.csv'
                #layer = 'params-module.positional_embedding.csv' # YES!
                #layer = 'params-module.text_projection.csv' # NO!
                layer = 'params-module.token_embedding.weight.csv'
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/open_clip/{file}/data/{i}/{layer}', names=list(range(13)))
                df = proc(df, lim)
                ax.plot(df.iloc[:, 0], df.iloc[:, 4], color=color)
                #ax.plot(df.iloc[:, 0], df.iloc[:, 5], color=color, alpha=0.6)
                ax.plot(df.iloc[:, 0], df.iloc[:, 6], color=color, alpha=0.3)
                ax.set_yscale('log')
                ax.set_ylabel('token_embedding grad mean and max')



    for j, ax in enumerate(axlist):
        if j == 0:
            ax.legend()
        ax.grid()
        ax.set_xlabel('Iterations')
        vv = 3787
        dd = 1e2
        # vv = 2186
        # dd = 1e3
        vv =8010
        dd = 1e1
        continue
        ax.axvline(vv, linestyle='--', color='gray')
        ax.set_xlim(vv-dd, vv+dd)

    plt.savefig('plots/loss_plot.png', bbox_inches='tight')



"""
step,
p.pow(2).sum().pow(0.5).item(), #'weight_norms'
p.abs().max().item(), # 'weight_maxs'
p.grad.pow(2).sum().pow(0.5).item(), # 'grad_norms'
p.grad.abs().max().item(), # 'grad_maxs'
optimizer.state[p]['exp_avg'].pow(2).sum().pow(0.5).item(), # 'exp_avgs_norms'
optimizer.state[p]['exp_avg'].abs().max().item(), # 'exp_avgs_maxs'
optimizer.state[p]['exp_avg_sq'].pow(2).sum().pow(0.5).item(), # 'exp_avg_sqs_norms'
optimizer.state[p]['exp_avg_sq'].abs().max().item(), # 'exp_avg_sqs_maxs'
"""

"""
0 step,
1 p.abs().mean().item(), #'weight_means'
2 p.abs().std().item(), # 'weight_std'
3 p.abs().max().item(), # 'weight_max'
4 p.grad.abs().mean().item(), #'grad_means'
5 p.grad.abs().std().item(), # 'grad_std'
6 p.grad.abs().max().item(), # 'grad_max'
optimizer.state[p]['exp_avg'].abs().mean().item(), #'v_means'
optimizer.state[p]['exp_avg'].abs().std().item(), # 'v_std'
optimizer.state[p]['exp_avg'].abs().max().item(), # 'v_max'
optimizer.state[p]['exp_avg_sq'].abs().mean().item(), #'g_means'
optimizer.state[p]['exp_avg_sq'].abs().std().item(), # 'g_std'
optimizer.state[p]['exp_avg_sq'].abs().max().item(), # 'g_max'
"""

"""
        features = x.abs()
        to_log = [
            _iter,
            features.std().item(), # std
            features.mean().item(), # mean
            features.max().item(), # max
        ]
"""