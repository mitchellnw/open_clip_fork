import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


#plt.rcParams['font.family'] = 'STIXGeneral'

cmap=plt.get_cmap('cool')
def get_metrics(filename):
    if not os.path.exists(filename):
        return 1./1000
    f = open(filename, "r")
    c = f.read()
    good =[l for l in c.split("\n") if "imagenet-zero" in l]
    if len(good) != 1:
        return 100 * 1./1000
    top5 = float(good[0].split("\t")[1].split(" ")[-1])
    top1 = float(good[0].split("\t")[0].split(" ")[-1])
    return 100 * top1

def proc(df, lim=-1):
    df.drop(df[df[0] == 0].index[1:], axis=0, inplace=True)
    df.drop_duplicates(0, keep='last', inplace=True)
    if lim > 0:
        df = df[df[0] < lim]
    return df
if __name__ == '__main__':
    kernel_size = 40
    min_loss = 14
    max_scaler = 1
    log_level = 1  #+ len(modules)

    # NOTE: LOOK AT FEATURE STDDEV!

    #fig, axlist = plt.subplots(log_level, 2, figsize=(16, 5 * log_level))
    fig = plt.figure(figsize=(14, 3))#, layout='tight')
    gs = gridspec.GridSpec(1, 3)
    # if log_level == 1:
    #     axlist = [axlist]

    axlabels = []

    convert = {
        'B-32' : 'Base',
        'L-14' : 'Large',
        'H-14' : 'Huge',
    }

    #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
    for k, model in enumerate(['B-16', 'L-16', 'H-14']):
        ax = fig.add_subplot(gs[0, k])
        axins2 = zoomed_inset_axes(ax, zoom=4, loc=1)

        for template, name, color, marker in [
            ('cat-ViT-{}-pd05-65536-1e-3-0.99-v0', 'b/32 - 995',cmap(1.), -1),
            ('cat-ViT-{}-pd05-65536-1e-3-0.98-v0', 'b/32 - 995',cmap(3./4), -1),
            ('cat-ViT-{}-pd05-65536-1e-3-0.95-v0', 'b/32 - 995',cmap(2./4), -1),
            ('cat-ViT-{}-pd05-65536-1e-3-0.9-v0', 'b/32 - 995',cmap(1./4), -1),
            ('cat-ViT-{}-pd05-65536-1e-3-0.8-v0', 'b/32 - 995',cmap(0./4), -1),
        ]:
            newtemp = template.format(model)
            fname = f'/fsx/home-mitchellw/experimetns/b2cpbackup/{newtemp}/data/0/loss.csv'
            if not os.path.exists(fname):
                fname = fname.replace('v0', 'v1')
            if not os.path.exists(fname):
                continue
            df = pd.read_csv(fname, names=list(range(2)))
            df = proc(df, -1)
            #df = df[df[0] > 30000]
            #ax.set_yscale('log')
            #ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), color=color,alpha=1, label=name)#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
            
            kernel = np.ones(kernel_size) / kernel_size
            data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
            data_convolved = data_convolved[kernel_size:-kernel_size]
            ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=2)
            axins2.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, linewidth=2)
            axins2.set_xlim(34000, 36000)
            if k >= 1:
                axins2.set_ylim([0.25, 1.])
            else:
                axins2.set_ylim([0.5, 1.5])

        ax.set_xlabel('Iteration', fontsize=14)
        ax.set_ylabel('Loss', fontsize=14)
        if k == 1:
            leg = ax.legend(bbox_to_anchor=(0.6 + 0.15, -0.27), ncol=2,fontsize=11)
            # leg.get_texts()[-1].set_fontweight('bold')


        axins2.set_xticks([])
        axins2.set_yticks([])
        axins2.tick_params(labelleft=False, labelbottom=False)
        mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.2")

        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid()
        #ax.set_ylim([1-k,6])


    if k == 0:
        ax.set_title("ViT-Base model", fontsize=12)
    else:
        ax.set_title("ViT-Huge model", fontsize=12)

    # for k, model in enumerate(['B-32', 'H-14']):
    #     ax = fig.add_subplot(gs[0, k+2])
    #     axins2 = zoomed_inset_axes(ax, zoom=4, loc=1)


    #     for template, name, color, marker in [

    #         ('clipadamw-ViT-{}-16384-2e-3-0.98-v0', 'bfloat16 baseline','C0', 's'),
    #         ('clipadamw-camp65kfp8global-ViT-{}-16384-2e-3-0.98-v0', 'standard fp8 baseline','C5', '^'),
    #         ('clipadamw-camp65kfp8mix-ViT-{}-16384-2e-3-0.98-v0', 'SwitchBack fp8','C6', 'o'),
    #     ]:
    #         newtemp = template.format(model)
    #         fname = f'/fsx/home-mitchellw/experimetns/b2cpbackup/{newtemp}/data/0/loss.csv'
    #         df = pd.read_csv(fname, names=list(range(2)))
    #         df = proc(df, 9700 if k == 1 and 'global' in newtemp else -1)
    #         #df = df[df[0] > 30000]
    #         #ax.set_yscale('log')
    #         #ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), color=color,alpha=1, label=name)#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
            
    #         kernel = np.ones(kernel_size) / kernel_size
    #         data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
    #         data_convolved = data_convolved[kernel_size:-kernel_size]
    #         ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=2)
    #         axins2.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, linewidth=2)
    #         axins2.set_xlim(18000, 20000)
    #         if k == 1:
    #             axins2.set_ylim([0.25, 1.])
    #         else:
    #             axins2.set_ylim([1.5, 2.25])
    #         # print(file)
    #     #leg = ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.3))
    #     ax.set_xlabel('Iteration', fontsize=14)
    #     ax.set_ylabel('Loss', fontsize=14)
    #     if k == 1:
    #         leg = ax.legend(bbox_to_anchor=(0.65 + 0.15, -0.27), ncol=2, fontsize=11)
    #         leg.get_texts()[-1].set_fontweight('bold')

    #     axins2.set_xticks([])
    #     axins2.set_yticks([])
    #     axins2.tick_params(labelleft=False, labelbottom=False)
    #     mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.2")

    #     ax.tick_params(axis='x', labelsize=11)
    #     ax.tick_params(axis='y', labelsize=11)
    #     ax.grid()
    #     ax.set_ylim([1-k,6])

    #     if k == 0:
    #         ax.set_title("ViT-Base model", fontsize=12)
    #     else:
    #         ax.set_title("ViT-Huge model", fontsize=12)
    fig.subplots_adjust(
        top=0.95, left=0.07, right=0.9, bottom=0.3, wspace=0.32, hspace=0.28
    )

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/paper2/beta2_plot_fb.png', bbox_inches='tight')

