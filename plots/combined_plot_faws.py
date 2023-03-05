import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import glob
import wandb

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

def get_metrics(sname):
    files = list(glob.glob("src/evals-faws/*.pt"))
    metrics = {}
    for filename in files:
        if 'latest' in filename:
            continue
        # get name
        name = filename.split('/')[-1].split('---')[0]
        if name not in metrics:
            metrics[name] = []
        epoch = int(filename.split("_")[-1].split(".")[0])
        f = open(filename, "r")
        c = f.read()
        good =[l for l in c.split("\n") if "imagenet-zero" in l]
        if len(good) != 1:
            continue
        top5 = float(good[0].split("\t")[1].split(" ")[-1])
        top1 = float(good[0].split("\t")[0].split(" ")[-1])
        metrics[name].append([epoch, top1, top5])

    for name in metrics:
        metrics[name].sort(key=lambda x:x[0])
        #print(name, metrics)
    return metrics[sname]




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
    log_level = 2

    # NOTE: LOOK AT FEATURE STDDEV!

    file_list = []
    file_list = [
        #('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-098-amp_bfloat16-v1', 'standard (lr=1e-3, beta2=0.98, warmup=8k, bs=32k)', 'C0', -1),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-098-amp_bfloat16-pinit-v1', 'p-init (lr=1e-3, beta2=0.98, warmup=8k, bs=32k)', 'C1', 5000),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-098-amp_bfloat16-cinit-v1', 'c-init (lr=1e-3, beta2=0.98, warmup=8k, bs=32k)', 'C3', -1),#3900),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-098-amp_bfloat16-pinit-cinit-v1', 'c-init + p-init (lr=1e-3, beta2=0.98, warmup=8k, bs=32k)', 'C2', -1),#3900),

        ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-rs-v1', 'standard (lr=1e-3, beta2=0.95, warmup=8k, bs=32k)', 'C0', -1),
        ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-pinit-rs-v1', 'p-init (lr=1e-3, beta2=0.95, warmup=8k, bs=32k)', 'C1', -1),
        ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-fixcinit-rs-v1', 'c-init (lr=1e-3, beta2=0.95, warmup=8k, bs=32k)', 'C2', -1),
        ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-pinit-fixcinit-rs-v1', 'p-init + c-init (lr=1e-3, beta2=0.95, warmup=8k, bs=32k)', 'C4', -1),

    ]
    #file_list = file_list[1:2]
    #file_list = file_list[:2]

    fig, axlist = plt.subplots(log_level, 1, figsize=(8, 5 * log_level))
    axins2 = zoomed_inset_axes(axlist[1], zoom=4, loc=4)
    if log_level == 1:
        axlist = [axlist]
    for j, (file, name, color, lim) in enumerate(file_list):

        if log_level >= 1:
            ax = axlist[0]
            for i in range(1):
                df = pd.read_csv(f'/fsx-labs/mitchellw/experiments/openclip2/{file}/data/{i}/loss.csv', names=list(range(2)))
                df = proc(df, lim)
                ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), color=color, alpha=0.2)#, label=name)# alpha=0.5,
                
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=1)
                ax.set_ylabel('Loss')
                #ax.set_yscale('log')
                #ax.set_xscale('log')

        
        if log_level >= 2:
            ax = axlist[1]
            
            for i in range(1):
                metrics = get_metrics('eval_' + file)
                print(color)
                if lim > 0:
                    metrics = metrics[:3]
                ax.plot([x[0] for x in metrics], [100*x[1] for x in metrics], color=color, marker='o')
                axins2.plot([x[0] for x in metrics], [100*x[1] for x in metrics], color=color, marker='o')
                axins2.set_xlim(10.5, 12.1)
                axins2.set_ylim(66, 67.1)
                ax.set_ylabel("Zero-shot ImageNet (top-1, %)")


    for j, ax in enumerate(axlist):
        if j == 0:
            ax.legend()
        ax.grid()
        ax.set_xlabel('Iterations')
        if j == 1:
            ax.set_xlabel('Checkpoint intervals')
        vv = 3787
        dd = 1e2
        # vv = 2186
        # dd = 1e3
        vv =8010
        dd = 1e1
        continue
        ax.axvline(vv, linestyle='--', color='gray')
        ax.set_xlim(vv-dd, vv+dd)

    axins2.set_xticks([])
    axins2.set_yticks([])

    axins2.tick_params(labelleft=False, labelbottom=False)
    mark_inset(ax, axins2, loc1=2, loc2=4, fc="none", ec="0.5")
    plt.savefig('plots/combined_plot_faws.png', bbox_inches='tight')



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