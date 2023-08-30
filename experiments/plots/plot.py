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

       

        to_enum = [
            ('cosine1b', 'CosineLR no avg, Data=1.3B, LR=1e-3',cmap(0.), -1),
        ]
        

        for template, name, color, marker in reversed(to_enum):
            xs, ys = [], []
            newtemp = template#.format(model)
            for ep in range(20):
                if not os.path.exists(f'/p/project/ccstdl/wortsman1/open_clip_fork/logs/{newtemp}/checkpoints/eval_epoch_{ep}.pt'):
                    print(f'/p/project/ccstdl/wortsman1/open_clip_fork/logs/{newtemp}/checkpoints/eval_epoch_{ep}.pt', 'error')
                    continue
                fname = f'/p/project/ccstdl/wortsman1/open_clip_fork/logs//{newtemp}/checkpoints/eval_epoch_{ep}.pt'
                top1 = get_metrics(fname)

                xs.append(ep)
                ys.append(top1)

            ax.plot(xs, ys, marker='o', color=color, label=name)
            print(ys)



        to_enum = [
            ('cosine1b', 'CosineLR poly_16_1, Data=1.3B, LR=1e-3',cmap(0.5), -1),
        ]
        

        for template, name, color, marker in reversed(to_enum):
            xs, ys = [], []
            newtemp = template#.format(model)
            for ep in range(20):
                if not os.path.exists(f'/p/project/ccstdl/wortsman1/open_clip_fork/logs/{newtemp}/checkpoints/eval_poly_16_1_{ep}.pt'):
                    print(f'/p/project/ccstdl/wortsman1/open_clip_fork/logs/{newtemp}/checkpoints/eval_poly_16_1_{ep}.pt', 'error')
                    continue
                fname = f'/p/project/ccstdl/wortsman1/open_clip_fork/logs//{newtemp}/checkpoints/eval_poly_16_1_{ep}.pt'
                top1 = get_metrics(fname)

                xs.append(ep)
                ys.append(top1)

            ax.plot(xs, ys, marker='o', color=color, label=name)
            print(ys)

        to_enum = [
            ('cosine1b', 'CosineLR poly_8_1, Data=1.3B, LR=1e-3',cmap(1.), -1),
        ]
        

        for template, name, color, marker in reversed(to_enum):
            xs, ys = [], []
            newtemp = template#.format(model)
            for ep in range(20):
                if not os.path.exists(f'/p/project/ccstdl/wortsman1/open_clip_fork/logs/{newtemp}/checkpoints/eval_poly_8_1_{ep}.pt'):
                    print(f'/p/project/ccstdl/wortsman1/open_clip_fork/logs/{newtemp}/checkpoints/eval_poly_8_1_{ep}.pt', 'error')
                    continue
                fname = f'/p/project/ccstdl/wortsman1/open_clip_fork/logs//{newtemp}/checkpoints/eval_poly_8_1_{ep}.pt'
                top1 = get_metrics(fname)

                xs.append(ep)
                ys.append(top1)

            ax.plot(xs, ys, marker='o', color=color, label=name)
            print(ys)

        # to_enum = [
        #     ('clipadamw-const_poly_8_1-ViT-B-32-16384-3e-4-0.99-100-v0', 'ConstantLR + Poly-8 Avg., LR=3e-4',cmap(0.), -1),
        #     ('clipadamw-const_poly_8_1-ViT-B-32-16384-5e-4-0.99-100-v0', 'ConstantLR + Poly-8 Avg., LR=5e-4',cmap(1./3), -1),
        #     ('clipadamw-const_poly_8_1-ViT-B-32-16384-1e-3-0.99-100-v0', 'ConstantLR + Poly-8 Avg., LR=1e-3',cmap(2./3), -1),
        #     ('clipadamw-const_poly_8_1-ViT-B-32-16384-2e-3-0.99-100-v0', 'ConstantLR + Poly-8 Avg., LR=2e-3',cmap(1.), -1),
        # ]
        

        # for template, name, color, marker in reversed(to_enum):
        #     xs, ys = [], []
        #     newtemp = template#.format(model)
        #     for ep in range(100):
        #         if not os.path.exists(f'/p/project/ccstdl/wortsman1/open_clip_fork/logs//{newtemp}/checkpoints/eval_ema_1_{ep}.pt'):
        #             continue
        #         fname = f'/p/project/ccstdl/wortsman1/open_clip_fork/logs//{newtemp}/checkpoints/eval_ema_1_{ep}.pt'
        #         top1 = get_metrics(fname)
        #         if top1 < 0.5:
        #             continue

        #         xs.append(ep)
        #         ys.append(top1)

        #     ax.plot(xs, ys, marker='^', color=color, label=name, linestyle='--')

        # to_enum = [
        #     ('clipadamw-const_poly_8_1-ViT-B-32-16384-5e-4-0.99-100-v0', 'ConstantLR, LR=5e-4',cmap(0.), -1),
        #     ('clipadamw-const_poly_8_1-ViT-B-32-16384-1e-3-0.99-100-v0', 'ConstantLR, LR=1e-3',cmap(0.5), -1),
        #     ('clipadamw-const_poly_8_1-ViT-B-32-16384-2e-3-0.99-100-v0', 'ConstantLR, LR=2e-3',cmap(1.), -1),
        # ]
        

        # for template, name, color, marker in reversed(to_enum):
        #     xs, ys = [], []
        #     newtemp = template#.format(model)
        #     for ep in range(100):
        #         if not os.path.exists(f'/p/project/ccstdl/wortsman1/open_clip_fork/logs//{newtemp}/checkpoints/eval_epoch_{ep}.pt'):
        #             continue
        #         fname = f'/p/project/ccstdl/wortsman1/open_clip_fork/logs//{newtemp}/checkpoints/eval_epoch_{ep}.pt'
        #         top1 = get_metrics(fname)
        #         if top1 < 0.5:
        #             continue

        #         xs.append(ep)
        #         ys.append(top1)

        #     ax.plot(xs, ys, marker='d', color=color, label=name, linestyle=':')



    ax.set_xlabel('Virtual epoch', fontsize=12)
    ax.set_ylabel('Zero-shot ImageNet', fontsize=12)



    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.grid()
    #ax.set_ylim([1-k,6])



    # ax.set_ylabel('Zero-shot ImageNet accuracy', fontsize=12)
    # ax.set_xlabel('Beta2', fontsize=12)
    leg = ax.legend(fontsize=10,  bbox_to_anchor=(0.8, -0.3),)
    #ax.set_title('ViT-Huge', fontsize=10)
    fig.subplots_adjust(
        top=0.95, left=0.07, right=0.9, bottom=0.3, wspace=0.32, hspace=0.28
    )

    plt.savefig('/p/project/ccstdl/wortsman1/open_clip_fork/experiments/plots/plot.png', bbox_inches='tight')