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

    ys1 = [66.57, 75.25, 77.94, 80.1]
    ys2 = [66.31, 75.16, 77.63, 79.52]
    ys3 = [65.70, 74.97, 77.65, 78.19]
    ax.plot(ys1, color='C0', label='bfloat16', marker='s', markersize=7)
    ax.plot(ys2, color='C2', label='switchback (vectorize)', marker='^', markersize=7)
    ax.plot(ys3, color='C4', label='switchback', marker='o', markersize=7)

    sizes = ['Basee', 'Large', 'Huge', 'Giant']
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



    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/paper/inference_quant.png', bbox_inches='tight')

