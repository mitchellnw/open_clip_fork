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
    fig = plt.figure(figsize=(14, 4 * 3))#, layout='tight')
    gs = gridspec.GridSpec(4, 2)
    # if log_level == 1:
    #     axlist = [axlist]
    firsttime = True
    axlabels = []

    convert = {
        'B-32' : 'Base',
        'L-14' : 'Large',
        'H-14' : 'Huge',
    }

    for ii, mm in enumerate([
        'opt3/clipadamw-ViT-H-14-16384-2e-3-0.98-v0',
        'opt3/clipadamw-ViTls0-H-14-16384-2e-3-0.98-v0',
        'opt3/clipadamw-ViT-L-14-16384-2e-3-0.98-v0',
        'opt3/clipadamw-ViTls0-L-14-16384-2e-3-0.98-v0',
    ]):

        ax = fig.add_subplot(gs[ii, 0])
        ax.grid()
        #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
        if ii == 0:
            ax.set_title('ViT-Huge, no layer scale')
        if ii == 1:
            ax.set_title('ViT-Huge, layer scale init 0')
        if ii == 2:
            ax.set_title('ViT-Large, no layer scale')
        if ii == 3:
            ax.set_title('ViT-Large, layer scale init 0')
        for k, (model, d) in enumerate([

            #('loss', 1),
            #('params-module.visual.transformer.resblocks.0.attn.in_proj_weight', 14),

            #('params-module.visual.conv1.weight', 14),
            #('params-module.visual.conv1.weight', 6),
            ('params-module.visual.transformer.resblocks.0.mlp.c_proj.weight', 6),
            ('params-module.visual.transformer.resblocks.10.mlp.c_proj.weight', 6),
            ('params-module.visual.transformer.resblocks.20.mlp.c_proj.weight', 6),
            ('params-module.visual.transformer.resblocks.30.mlp.c_proj.weight', 6),

        ]):

            for template, name, color, marker in [
                #('opt3/clipadamw-ViT-H-14-16384-2e-3-0.98-v1', 'gc', 'C6', -1),
                (mm, 'gc', 'C6', -1),

            ]:
                newtemp = template.format(model)
                fname = f'/fsx/home-mitchellw/experimetns/{newtemp}/data/0/{model}.csv'
                if not os.path.exists(fname):
                    print('ERROR', fname)
                    continue
                df = pd.read_csv(fname, names=list(range(25)))
                df = proc(df, -1)


                ax.set_yscale('log')

                tdf = df# df[(df[0] > l1) & (df[0] < l2)]

                # ax.plot(tdf.iloc[:, 0], tdf.iloc[:, 4], color=f'C{k}',label=name, linewidth=0.9, alpha=0.5)#, marker='o')#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
                # ax.plot(tdf.iloc[:, 0], tdf.iloc[:, 6], color=f'C{k}',alpha=1, label=name, linewidth=0.9)#, marker='o')#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,


                kernel_size = 40
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(np.sqrt(tdf.iloc[:, 6]), kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(tdf.iloc[:, 0][kernel_size:-kernel_size], data_convolved, color=f'C{k}', alpha=1, linewidth=2, label=f'Block {model[-2:]} max')



                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(np.sqrt(tdf.iloc[:, 4]), kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(tdf.iloc[:, 0][kernel_size:-kernel_size], data_convolved, color=f'C{k}', linewidth=2, label=f'Block {model[-2:]} max', alpha=0.4)


            ax.set_ylim([1e-9, 2e-2])
            ax.set_ylabel('MLP weight gradient', fontsize=12)
            ax.set_xlabel('Iteration', fontsize=12)


        ax = fig.add_subplot(gs[ii, 1])
        if ii == 0:
            ax.set_title('ViT-Huge, no layer scale')
        if ii == 1:
            ax.set_title('ViT-Huge, layer scale init 0')
        if ii == 2:
            ax.set_title('ViT-Large, no layer scale')
        if ii == 3:
            ax.set_title('ViT-Large, layer scale init 0')
        ax.grid()
        #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
        for k, (model, d) in enumerate([

            #('loss', 1),
            #('params-module.visual.transformer.resblocks.0.attn.in_proj_weight', 14),

            #('params-module.visual.conv1.weight', 14),
            ('features2-module.visual.transformer.resblocks.0', 3),
            ('features2-module.visual.transformer.resblocks.10', 3),
            ('features2-module.visual.transformer.resblocks.20', 3),
            ('features2-module.visual.transformer.resblocks.30', 3),

        ]):

            for template, name, color, marker in [
                #('opt3/clipadamw-ViT-H-14-16384-2e-3-0.98-v1', 'gc', 'C6', -1),
                (mm, 'gc', 'C6', -1),
            ]:
                newtemp = template.format(model)
                fname = f'/fsx/home-mitchellw/experimetns/{newtemp}/data/0/{model}.csv'
                if not os.path.exists(fname):
                    print('ERROR', fname)
                    continue
                df = pd.read_csv(fname, names=list(range(4)))
                df = proc(df, -1)


                ax.set_yscale('log')

                tdf = df# df[(df[0] > l1) & (df[0] < l2)]

                # ax.plot(tdf.iloc[:, 0], tdf.iloc[:, 2], color=f'C{k}', linewidth=0.9, alpha=0.5, label=f'Block {model[-2:]} mean') 
                # ax.plot(tdf.iloc[:, 0], tdf.iloc[:, 3], color=f'C{k}', alpha=1, linewidth=0.9, label=f'Block {model[-2:]} max')


                kernel_size = 40
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(tdf.iloc[:, 3], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(tdf.iloc[:, 0][kernel_size:-kernel_size], data_convolved, color=f'C{k}', alpha=1, linewidth=2, label=f'Block {model[-2:]} max')



                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(tdf.iloc[:, 2], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(tdf.iloc[:, 0][kernel_size:-kernel_size], data_convolved, color=f'C{k}', linewidth=2, label=f'Block {model[-2:]} max', alpha=0.4)




            ax.set_ylim([0.05, 2000])
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Transformer block output', fontsize=12)

            if firsttime:
                myax = ax
                firsttime = False




                


        # if k == 1:
        #     ax.set_xlabel('    Iteration', fontsize=11)
        #     ax.set_yticks([1,2,3,4])
        #     ax.set_yticklabels(['1', '2', '3', '4'])
        # if model == 'loss':
        #     ax.set_ylabel('Loss', fontsize=11)
        # elif model == 'amp':
        #     ax.set_yscale('log', base=2)
        #     ax.set_ylabel('Grad scaler', fontsize=11)
        # elif d == 6:
        #     ax.set_ylabel('Grad absmax', fontsize=11)
        #     ax.set_title(model.replace('params-module.', ''), fontsize=9, loc='left', y=1.1, pad=-20, x = 0.01)

        # elif d == 14:
        #     ax.set_ylabel('RMS', fontsize=11)
        #     ax.set_title(model.replace('params-module.', ''), fontsize=9, loc='left', y=1.1, pad=-20, x = 0.01)

        # elif d == 3:
        #     ax.set_ylabel('Feature max', fontsize=11)
        #     ax.set_title(model.replace('features2-module.', ''), fontsize=9, loc='left', y=1.1, pad=-20, x = 0.01)

        # ax.tick_params(axis='x', labelsize=10)
        # ax.tick_params(axis='y', labelsize=10)
        # ax.grid()
    import matplotlib.lines as mlines

    # red_dashed_line = mlines.Line2D([], [], color='red', linestyle='--', label='RMS spike in embedding layer which preceeds loss spike')

    # grey_dashed_line = mlines.Line2D([], [], color='grey', linestyle='--', label='Inf gradient causing grad scaler decrease')

    # Add the custom legend to the plot
    #ax.legend(handles=[red_dashed_line, grey_dashed_line], bbox_to_anchor=(0.5, -0.27), ncol=2,fontsize=10.5)

    # red_dashed_line = mlines.Line2D([], [], color='C9', linestyle='--', label='RMS spike preceeding loss spike')
    # red_dashed_line1 = mlines.Line2D([], [], color='C6', linestyle='-', label='beta2 = 0.98')
    # red_dashed_line2 = mlines.Line2D([], [], color='gray', linestyle='-', label='beta2 = 0.9')
    #ax.legend(handles=[red_dashed_line, red_dashed_line1, red_dashed_line2], bbox_to_anchor=(0.5025, 0.85), ncol=3,fontsize=10)
    myax.legend( bbox_to_anchor=(0.53, -4.4), ncol=4,fontsize=10.5)

    fig.subplots_adjust(
        top=0.95, left=0.07, right=0.9, bottom=0.05, wspace=0.2, hspace=0.35
    )

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/paper3/compare_amp_2.pdf', bbox_inches='tight')
