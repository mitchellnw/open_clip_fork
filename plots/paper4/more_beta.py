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
    fig = plt.figure(figsize=(12, 3))#, layout='tight')
    gs = gridspec.GridSpec(1, 2)
    # if log_level == 1:
    #     axlist = [axlist]

    axlabels = []

    convert = {
        'B-32' : 'Base',
        'L-14' : 'Large',
        'H-14' : 'Huge',
    }
    idx = {
        0.45 : -.2,
        0.5 : 0,
        0.65 : 0.5,
        0.8 : 1,

        0.9 : 2,
        0.95 : 3,
        0.98 : 4,
        0.99 : 5,
        0.995 : 6,
    }

    #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
    for k, model in enumerate(['H-14']):
        ax = fig.add_subplot(gs[0, k])
        #axins2 = zoomed_inset_axes(ax, zoom=10 if k ==2 else 8, loc=1)

        xs, ys = [], []

        to_enum = [
            ('customadamw-ViT-{}-16384-2e-3-0.99-v1', 'beta2 = 0.99',cmap(1.), -1),
            ('customadamw-ViT-{}-16384-2e-3-0.98-v1', 'beta2 = 0.98',cmap(3./4), -1),
            ('customadamw-ViT-{}-16384-2e-3-0.95-v1', 'beta2 = 0.95',cmap(2./4), -1),
            ('customadamw-ViT-{}-16384-2e-3-0.9-v1', 'beta2 = 0.9',cmap(1./4), -1),
            ('customadamw-ViT-{}-16384-2e-3-0.8-v4', 'beta2 = 0.8',cmap(1./4), -1),

            #('customadamw-ViT-{}-16384-2e-3-0.8-v1', 'beta2 = 0.8','k', -1),
            ('customadamw-ViT-{}-16384-2e-3-0.5-v1', 'beta2 = 0.5',cmap(0./4), -1),
        ]
        

        # for template, name, color, marker in reversed(to_enum):
        #     newtemp = template.format(model)
        #     if not os.path.exists(f'/fsx/home-mitchellw/experimetns/opt/{newtemp}/checkpoints/eval.pt'):
        #         newtemp = newtemp.replace('v1', 'v0')
        #     if not os.path.exists(f'/fsx/home-mitchellw/experimetns/opt/{newtemp}/checkpoints/eval.pt'):
        #         print('error', newtemp)
        #     fname = f'/fsx/home-mitchellw/experimetns/opt/{newtemp}/checkpoints/eval.pt'
        #     top1 = get_metrics(fname)
        #     print(fname, top1)
        #     if top1 > 0.1:
        #         xs.append(float(name.split("=")[-1].strip()))
        #         ys.append(top1)

        # ax.plot([idx[j] for j in xs], ys, marker='o', color=color, label='default', markersize=6.5)


        #     # ('opt3/customadamw-ViT-{}-16384-2e-3-0.99-gc-v0', '+ grad clipping', 'C9', -1),
        #     # ('opt3/clipadamw-ViT-{}-16384-2e-3-0.99-v0', '+ update clipping','C1', -1),
        
        # xs, ys = [], []
        # to_enum = [
        #     ('opt3/customadamw-ViT-{}-16384-2e-3-0.995-gc-v0','beta2 = 0.995', cmap(0.), -1),
        #     ('opt3/customadamw-ViT-{}-16384-2e-3-0.99-gc-v0','beta2 = 0.99', cmap(0.), -1),
        #     ('opt3/customadamw-ViT-{}-16384-2e-3-0.98-gc1-v0','beta2 = 0.98', cmap(0.), -1),
        #     ('opt3/customadamw-ViT-{}-16384-2e-3-0.95-gc-v0','beta2 = 0.95', cmap(0.), -1),

        # ]

        # for template, name, color, marker in reversed(to_enum):
        #     newtemp = template.format(model)
        #     fname = f'/fsx/home-mitchellw/experimetns/{newtemp}/checkpoints/eval.pt'
        #     top1 = get_metrics(fname)
        #     print(fname, top1)
        #     if top1 > 0.1:
        #         xs.append(float(name.split("=")[-1].strip()))
        #         ys.append(top1)
        # ax.plot([idx[j] for j in xs], ys, marker='s', color=color, label='+ grad clipping', markersize=6.5)


        xs, ys = [], []
        to_enum = [
            #('clipadamw-ViT-{}-16384-2e-3-0.995-v0','beta2 = 0.99', cmap(0.5), -1),
            ('opt3/clipadamw-ViT-{}-16384-2e-3-0.995-v4','beta2 = 0.995', 'C1', -1),

            ('opt3/clipadamw-ViT-{}-16384-2e-3-0.99-v0','beta2 = 0.99', 'C1', -1),
            ('opt3/clipadamw-ViT-{}-16384-2e-3-0.98-v0','beta2 = 0.98', 'C1', -1),
            #('opt3/clipadamw-ViT-{}-16384-2e-3-0.95-v4','beta2 = 0.95', 'C1', -1),

        ]

        for template, name, color, marker in reversed(to_enum):
            newtemp = template.format(model)
            fname = f'/fsx/home-mitchellw/experimetns/{newtemp}/checkpoints/eval.pt'
            top1 = get_metrics(fname)
            print(fname, top1)
            if top1 > 0.1:
                xs.append(float(name.split("=")[-1].strip()))
                ys.append(top1)
        ax.plot(xs, ys, marker='^', color=color, label='+ update clipping', markersize=6.5)


        # xs, ys = [], []
        # to_enum = [
        #     #('clipadamw-ViT-{}-16384-2e-3-0.995-v0','beta2 = 0.99', cmap(0.5), -1),
        #     ('opt3/clipadamw-ViTDP-{}-16384-2e-3-0.995-v0','beta2 = 0.995', 'C2', -1),
        #     ('opt3/clipadamw-ViTDP-{}-16384-2e-3-0.99-v0','beta2 = 0.99', 'C2', -1),
        #     #('opt/clipadamw-amp-ViTDP-{}-16384-2e-3-0.99-v1','beta2 = 0.99', 'C2', -1),
        #     ('opt/clipadamw-amp-ViTDP-{}-16384-2e-3-0.98-v1','beta2 = 0.98', 'C2', -1),
        #     #('opt/clipadamw-amp-ViTDP-{}-16384-2e-3-0.995-v1','beta2 = 0.9#', 'C2', -1),
        #     ('opt3/clipadamw-ViTDP-{}-16384-2e-3-0.95-v0','beta2 = 0.95', 'C2', -1),

        # ]

        # for template, name, color, marker in reversed(to_enum):
        #     newtemp = template.format(model)
        #     fname = f'/fsx/home-mitchellw/experimetns/{newtemp}/checkpoints/eval.pt'
        #     top1 = get_metrics(fname)
        #     print(fname, top1)
        #     if top1 > 0.1:
        #         xs.append(float(name.split("=")[-1].strip()))
        #         ys.append(top1)
        # ax.plot([idx[j] for j in xs], ys, marker='P', color=color, label='+ update clipping + input norm', markersize=7)



        xs, ys = [], []
        to_enum = [
            #('opt3/wclipadamw-ViT-{}-16384-2e-3-0.8-v0',f'beta2 = {1-20000**(-0.8)}', 'gray', -1),
            ('opt3/wclipadamw-ViT-{}-16384-2e-3-0.65-v0',f'beta2 = {1-20000**(-0.65)}', 'gray', -1),

            ('opt3/wclipadamw-ViT-{}-16384-2e-3-0.5-v0',f'beta2 = {1-20000**(-0.5)}', 'gray', -1),
            ('opt3/wclipadamw-ViT-{}-16384-2e-3-0.45-v0',f'beta2 = {1-20000**(-0.45)}', 'gray', -1),


        ]

        for template, name, color, marker in reversed(to_enum):
            newtemp = template.format(model)
            fname = f'/fsx/home-mitchellw/experimetns/{newtemp}/checkpoints/eval.pt'
            top1 = get_metrics(fname)
            print(fname, top1)
            if top1 > 0.1:
                xs.append(float(name.split("=")[-1].strip()))
                ys.append(top1)
        ax.plot(xs, ys, marker='H', color=color, label='+ update clipping + beta2 warmup', markersize=7)




        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)




        # # ax.set_xticks(idx.keys())
        ax.set_xticks([0.98, 0.985, 0.99, 0.995])
        ax.set_xticklabels([0.98, 0.985, 0.99, 0.995])

        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        ax.grid()
        #ax.set_ylim([1-k,6])


        # if k == 0:
        #     ax.set_title("ViT-Base model", fontsize=12)
        # elif k == 1:
        #     ax.set_title("ViT-Large model", fontsize=12)
        # elif k == 2:
        #     ax.set_title("ViT-Huge model", fontsize=12)
    ax.set_ylim([55, 58])
    ax.set_ylabel('Zero-shot ImageNet accuracy', fontsize=12)
    ax.set_xlabel('Beta2', fontsize=12)
    leg = ax.legend(fontsize=10)
    ax.set_title('ViT-Huge', fontsize=10)
    fig.subplots_adjust(
        top=0.95, left=0.07, right=0.9, bottom=0.3, wspace=0.32, hspace=0.28
    )

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/paper4/more_beta.pdf', bbox_inches='tight')
