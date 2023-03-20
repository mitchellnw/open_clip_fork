import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

import matplotlib.gridspec as gridspec

# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


#plt.rcParams['font.family'] = 'STIXGeneral'

cmap=plt.get_cmap('cool')
def get_metrics(filename):
    if not os.path.exists(filename):
        print(filename)
        return 100 * 1./1000
    f = open(filename, "r")
    c = f.read()
    good =[l for l in c.split("\n") if "imagenet-zero" in l]
    if len(good) != 1:
        return 100 * 1./1000
    top5 = float(good[0].split("\t")[1].split(" ")[-1])
    top1 = float(good[0].split("\t")[0].split(" ")[-1])
    return 100 * top1

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
    log_level = 1  #+ len(modules)

    # NOTE: LOOK AT FEATURE STDDEV!

    #fig, axlist = plt.subplots(log_level, 2, figsize=(16, 5 * log_level))
    fig = plt.figure(tight_layout=True, figsize=(10, 3.5))
    gs = gridspec.GridSpec(1, 2)
    # if log_level == 1:
    #     axlist = [axlist]

    axlabels = []

    convert = {
        'B-32' : 'Base',
        'L-14' : 'Large',
        'H-14' : 'Huge',
    }

    #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
    ax = fig.add_subplot(gs[0, 0])

    for template, name, color, marker in [
        ('clipadamw-ViT-H-14-16384-2e-3-0.98-v0', 'standard at init','C0', 's'),
        #('clipadamw-ViT-L-14-16384-2e-3-0.98-v0', 'L/14','C1', 's'),
        ('clipadamw-ViT-H-14-16384-2e-3-0.98-v0', 'standard final','C0', 's'),

        #('clipadamw-ViTls0-B-32-16384-2e-3-0.98-v0', 'B/32','C4', 's'),
        ('clipadamw-ViTls0-H-14-16384-2e-3-0.98-v0', 'zero-init layer scale at init','gray', 'o'),
        ('clipadamw-ViTls0-H-14-16384-2e-3-0.98-v0', 'zero-init layer scale final','gray', 'o'),

        # ('clipadamw-int8-ViT-{}-16384-2e-3-0.98-v0', 'LLM.int8() baseline','C1', '^'),
        # ('clipadamw-int8mix-ViT-{}-16384-2e-3-0.98-v0', 'SwitchBack int8','C4', 'o'),
    ]:
            
        xs, ys = [], []
        sizes = ['0', '10', '20', '30']

        for j, beta2 in enumerate(sizes):

            layer = f'features2-module.visual.transformer.resblocks.{beta2}.csv'
            filename = f'/fsx/home-mitchellw/experimetns/opt3/{template}/data/{0}/{layer}'
            if not os.path.exists(filename):
                continue
                

            df = pd.read_csv(filename, names=list(range(4)))
            if ' init' in name:
                top1 = df[2].values[0]
                ls = ':'
            else:
                top1 = df[2].values[-1]
                ls = '-'
            #if top1 > 0.1:
            xs.append(j)
            ys.append(top1)
        ax.plot(xs, ys, color=color, label=name, marker=marker, markersize=7 if marker=='s' else 7, linestyle=ls)
    ax.set_xticks([j for j, _ in enumerate(sizes)])
    ax.set_xticklabels(sizes)
    ax.set_xlabel('Transformer block index', fontsize=13)
    ax.set_ylabel('Average feature magnitude', fontsize=13)
    #leg = ax.legend(loc='center right', bbox_to_anchor=(1.0, 0.3))
    ax.legend()
    #leg.get_texts()[-1].set_fontweight('bold')

    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid()



    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/paper/feat_scaling.pdf', bbox_inches='tight')

