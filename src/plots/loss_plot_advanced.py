import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    kernel_size = 40
    min_loss = 10
    max_scaler = 0.25
    log_level = 4

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
    file_list = []
    file_list = ['h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v0', 'h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1', 'h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v2', 'h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1-try2']
    file_list = [file_list[1], file_list[-1]]
    file_list.append('l14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1')
    file_list.append('b16-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1')
    #file_list = [file_list[1], file_list[-1]]
    #file_list = [file_list[2]]
    fig, axlist = plt.subplots(log_level, 1, figsize=(8, 5 * log_level))
    for j, file in enumerate(file_list):

        if log_level >= 1:
            ax = axlist[0]
            for i in range(1):
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/loss.csv')
                ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), alpha=0.2, color=f'C{j}')
                
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=f'C{j}')

            #ax1.set_ylim([0, 2])
        if log_level >= 2:
            ax = axlist[1]
            for i in range(1):
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/amp.csv')
                ax.plot(df.iloc[:, 0], np.maximum(max_scaler, df.iloc[:, 1]))#, alpha=0.2)
                ax.set_yscale('log')

        if log_level >= 3:
            ax = axlist[2]
            for i in range(1):
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/params-module.logit_scale.csv')
                # idxs = df.iloc[:, 0]
                # ax.plot(df.iloc[:, 0], np.maximum(max_scaler, df.iloc[:, 1]))#, alpha=0.2)
                # ax.plot(df.iloc[:, 0], np.maximum(max_scaler, df.iloc[:, 2]))
                #ax.plot(df.iloc[:, 0], np.maximum(max_scaler, df.iloc[:, 3]))
                #ax.plot(df.iloc[:, 0], np.maximum(max_scaler, df.iloc[:, 1]))
                ax.plot(df.iloc[:, 0], df.iloc[:, 2], color=f'C{j}')
                #ax.plot(df.iloc[:, 0], df.iloc[:, 6], color=f'C{j}', linestyle='--', alpha=0.5)
                #ax.set_yscale('log')
                #ax.set_ylim([0.2, 0.3])
                ax.set_yscale('log')

        if log_level >= 4:
            ax = axlist[3]
            for i in range(1):
                df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/features-module.visual.transformer.resblocks.10.csv')

                
                if len(df.keys()) == 3:
                    xs = [ii // 2 for ii in range(len(df.index))]
                    df.insert(0, xs[0], xs)
                    print('correction')

                ax.plot(df.iloc[:, 0], df.iloc[:, 2], color=f'C{j}')
                ax.plot(df.iloc[:, 0], df.iloc[:, 3], color=f'C{j}', linestyle='--', alpha=0.8)
                #ax.fill_between(df.iloc[:, 0], df.iloc[:, 2], df.iloc[:, 3])
                ax.set_yscale('log')

    for ax in axlist:
        ax.grid()

    plt.savefig('plots/loss_plot_advanced.png', bbox_inches='tight')


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