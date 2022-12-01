import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    kernel_size = 40
    min_loss = 10
    max_scaler = 1
    log_level = 4

    # NOTE: LOOK AT FEATURE STDDEV!

    file_list = []
    file_list = ['clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v0', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v4', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v2']#, 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v3', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v4', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v5']
    file_list = ['clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v0', 'clip-h14-400m-l0-opt-0.001-0.9-0.98-1e-06-bs-8192-amp-v0']
    # file_list = [file_list[0], file_list[2], file_list[1], 'scps/clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8200-amp_bfloat16-v0/clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8200-amp_bfloat16-v0']#, file_list[4]]
    #file_list = [file_list[0]]
    #print(file_list)
    #file_list = [file_list[0], file_list[2], 'h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1-try2']
    #file_list = ['h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1-try2']
    # file_list.append('clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-4096-amp-v0')
    # file_list.append('clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-2048-amp-v1')
    #file_list.append('clip-l14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1')
    #file_list.append('clip-b16-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1')
    #file_list = [file_list[0], 'scps/clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8200-amp_bfloat16-v0/clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8200-amp_bfloat16-v0']
    #file_list = ['clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-4096-amp-v0', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-4096-amp-v1', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-4096-amp-v2', 'scps/clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-4080-amp_bfloat16-v0/clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-4080-amp_bfloat16-v0']
    #file_list = ['clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-2048-amp-v0', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-2048-amp-v1', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-2048-amp-v2']

    #file_list = ['clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v0', 'clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v2']
    fig, axlist = plt.subplots(log_level, 1, figsize=(8, 5 * log_level))
    for j, file in enumerate(file_list):

        if log_level >= 1:
            ax = axlist[0]
            for i in range(1):
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/loss.csv', names=list(range(2))).drop_duplicates(0, keep='last')
                ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), alpha=0.2, color=f'C{j}')
                
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=f'C{j}')
                ax.set_ylabel('Loss')

            #ax1.set_ylim([0, 2])
        if log_level >= 2:
            ax = axlist[1]
            for i in range(1):
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/amp.csv', names=list(range(2))).drop_duplicates(0, keep='last')
                ax.plot(df.iloc[:, 0], np.maximum(max_scaler, df.iloc[:, 1]))#, alpha=0.2)
                ax.set_yscale('log')
                ax.set_ylabel('Grad Scaler for Mixed Precision')
                stored_xlim = ax.get_xlim()
        
        if log_level >= 3:
            ax = axlist[2]
            for i in range(1):
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/features-module.visual.transformer.resblocks.30.csv', names=list(range(4))).drop_duplicates(0, keep='last')
                #df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/features-module.transformer.resblocks.20.csv', names=list(range(4)))#.drop_duplicates(0, keep='last')


                ax.plot(df.iloc[:, 0], df.iloc[:, 2], color=f'C{j}')
                ax.plot(df.iloc[:, 0], df.iloc[:, 3], color=f'C{j}', alpha=0.5)
                ax.set_yscale('log')
                ax.set_ylabel('Feature max (block 10)')
                ax.set_xlim(stored_xlim)
                

        #print(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/params-module.transformer.resblocks.30.attn.out_proj.weight.csv')
        if log_level >= 4:
            ax = axlist[3]
            for i in range(1):
                layer = 'params-module.visual.transformer.resblocks.0.mlp.c_fc.weight.csv'
                # 
                layer = 'params-module.logit_scale.csv'
                layer = 'params-module.positional_embedding.csv'
                layer = 'params-module.text_projection.csv'
                layer = 'params-module.token_embedding.weight.csv'
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/{layer}', names=list(range(13))).drop_duplicates(0, keep='last')
                ax.plot(df.iloc[:, 0], df.iloc[:, 10], color=f'C{j}')
                ax.plot(df.iloc[:, 0], df.iloc[:, 12], color=f'C{j}', alpha=0.5)
                ax.set_yscale('log')
                ax.set_ylabel('MLP-W Gradient Max (block 10)')
                ax.set_xlim(stored_xlim)



    for ax in axlist:
        ax.grid()
        ax.set_xlabel('Iterations')
        # ax.axvline(9978, linestyle='--', color='gray', alpha=0.5)
        # ax.set_xlim(9978 - 8, 9978 + 8)
        # ax.axvline(9889, linestyle='--', color='gray', alpha=0.5)
        # ax.set_xlim(9889 - 8, 9889 + 8)
        # ax.axvline(110434, linestyle='--', color='gray', alpha=0.5)
        #ax.set_xscale('log')
        # ax.axvline(48350, linestyle='--', color='gray', alpha=0.5)
        # ax.set_xlim(48350 - 10000, 48350 + 10000)
        # ax.axvline(71091, linestyle='--', color='gray', alpha=0.5)
        # ax.set_xlim(71091 - 2000, 71091 + 2000)
        # ax.axvline(69647, linestyle='--', color='gray', alpha=0.5)
        # ax.set_xlim(69647 - 1e1, 69647 + 1e1)
        # ax.axvline(92426, linestyle='--', color='gray', alpha=0.5)
        # ax.set_xlim(92426 - 1e1, 92426 + 1e1)
        # ax.axvline(8280, linestyle='--', color='gray', alpha=0.5)
        # ax.set_xlim(8280 - 1e1, 8280 + 1e1)
        # ax.set_xticks([int(j) for j in range(8280 - int(1e1), 8280 + int(1e1))])
        # ax.axvline(92426, linestyle='--', color='gray', alpha=0.5)
        # ax.set_xlim(94000 - 3e3, 94000 + 3e3)
    plt.savefig('plots/loss_plot_advanced.png', bbox_inches='tight')



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