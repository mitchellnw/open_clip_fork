import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.patches as patches
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
    fig = plt.figure(figsize=(14, 3 * 3))#, layout='tight')
    gs = gridspec.GridSpec(3, 3)
    # if log_level == 1:
    #     axlist = [axlist]

    axlabels = []

    convert = {
        'B-32' : 'Base',
        'L-14' : 'Large',
        'H-14' : 'Huge',
    }

    #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
    for k, (model, d) in enumerate([
        # 'params-module.visual.class_embedding',
        # 'params-module.visual.patch_embedding',

        #'params-module.visual.class_embedding',
        #'params-module.visual.conv1.weight',
        #params-module.transformer.resblocks.0..csv
        #'params-module.positional_embedding',

        # 'params-module.visual.conv1.weight',
        # 'params-module.visual.class_embedding',
        # 'params-module.visual.positional_embedding',

        ('loss', 1),
        ('amp', 1),
        #'params-module.visual.ln_pre.weight',
        # 'params-module.visual.ln_pre.bias',

        ('features2-module.visual.transformer.resblocks.20', 3),
        ('params-module.visual.conv1.weight', 14),
        #('params-module.visual.transformer.resblocks.0.attn.in_proj_weight', 14),
        

        ('params-module.visual.class_embedding', 6),
        # 'params-module.visual.positional_embedding',
        #'params-module.visual.transformer.resblocks.0.mlp.c_proj.weight',
        ('params-module.visual.transformer.resblocks.0.attn.in_proj_weight', 6),


        # 'params-module.visual.transformer.resblocks.0.ln_1.weight',
        # 'params-module.visual.transformer.resblocks.10.ln_1.weight',

        # 'params-module.visual.transformer.resblocks.0.mlp.c_fc.weight',
        # 'params-module.visual.transformer.resblocks.10.mlp.c_fc.weight',

    ]):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        #axins2 = zoomed_inset_axes(ax, zoom=4 if k ==2 else 8, loc=1)
        #params-module.transformer.resblocks.0.attn.in_proj_weight.csv
        for template, name, color, marker in [
            # ('opt3/clipadamw-ViT-H-14-16384-2e-3-0.98-v0', 'gc', 'C3', -1),
            # ('opt3/clipadamw-camp65k-ViT-H-14-16384-2e-3-0.98-v0', 'gc', 'C4', -1),
            #('opt3/clipadamw-amp-ViT-H-14-16384-2e-3-0.99-v0', 'gc', 'C5', -1),
            #('opt/customadamw-ViT-H-14-16384-2e-3-0.98-v4', 'gc', 'C6', -1),
            #('opt/customadamw-ViT-H-14-16384-2e-3-0.99-v4', 'gc', 'C5', -1),
            #('opt/customadamw-amp-ViT-H-14-16384-2e-3-0.98-extraln-v1', 'gc', 'C5', -1),
            ('opt/clipadamw-amp-ViTDP-L-14-16384-2e-3-0.98-v1', 'gc', 'C0', -1),

        ]:
            newtemp = template.format(model)
            fname = f'/fsx/home-mitchellw/experimetns/{newtemp}/data/0/{model}.csv'
            # if not os.path.exists(fname):
            #     fname = fname.replace('v0', 'v1')
            if not os.path.exists(fname):
                print('ERROR', fname)
                continue
            df = pd.read_csv(fname, names=list(range(25)))
            df = proc(df, -1)
            #df = df[df[0] > 30000]
            #ax.set_yscale('log')
            if d == 6:
                ax.set_yscale('log')

            l1 = 2900
            l2 = 3000
            l1 = 2879
            l2 = 3021
            ax.set_xlim(l1, l2)
            #axins2.set_xlim(l1, l2)
            # make an axvline wiht a skinny line
            ax.axvline(2924, color='red', linestyle='--', linewidth=0.9)
            #ax.axvline(2928, color='green', linestyle='--', linewidth=0.75)
            tdf = df[(df[0] > l1) & (df[0] < l2)]
            if d == 14:
                ax.plot(tdf.iloc[:, 0], np.sqrt(tdf.iloc[:, d]), color=color,alpha=1, label=name, linewidth=1.0)
            else:
                ax.plot(tdf.iloc[:, 0], np.sqrt(tdf.iloc[:, d]), color=color,alpha=1, label=name, linewidth=1.0)#, marker='o')#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
            badness = 0
            if d == 6:
                for axv in tdf[np.isinf(tdf[6])][0].values:
                    ax.axvline(axv, color='gray', linestyle='--', alpha=0.75, linewidth=0.9)
                    print(axv)
                    badness += 1

            



            #ax.set_xlim([3415, 3430])
        
            # kernel = np.ones(kernel_size) / kernel_size
            # data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
            # data_convolved = data_convolved[kernel_size:-kernel_size]
            # ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=2)

            #axins2.plot(tdf.iloc[:, 0], tdf.iloc[:, d], color=color,alpha=1)
            
            # if k == 0:
            #     axins2.set_ylim([0.4, 1.])
            # elif k == 2:
            #     axins2.set_xlim(19000, 20000)
            #     axins2.set_ylim([0.4, 0.6])
            # else:
            #     axins2.set_ylim([1.6, 1.95])

            # if 'amp' in model:
            #     ax.set_yscale('log')
            #     #axins2.set_yscale('log')

        ax.set_xlabel('Iteration', fontsize=11)
        if model == 'loss':
            ax.set_ylabel('Loss', fontsize=11)
        elif model == 'amp':
            ax.set_yscale('log', base=2)
            ax.set_ylabel('Loss scaler', fontsize=11)
        elif d == 6:
            ax.set_ylabel('Grad absmax', fontsize=11)
            #ax.set_title(model.replace('params-module.', ''), fontsize=9, loc='left', y=1.1, pad=-20, x = 0.01, zorder=2)
            title_text = model.replace('params-module.', '')
            textbox_props = dict(boxstyle='round, pad=0.3', facecolor='white', edgecolor='black', linewidth=1, alpha=0.8)
            ax.text(0.01, 0.974, title_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=textbox_props, zorder=2)
        elif d == 14:
            ax.set_ylabel('RMS', fontsize=11)
            #ax.set_title(model.replace('params-module.', ''), fontsize=9, loc='left', y=1.1, pad=-20, x = 0.01, zorder=2)
            title_text = model.replace('params-module.', '')
            textbox_props = dict(boxstyle='round, pad=0.3', facecolor='white', edgecolor='black', linewidth=1, alpha=0.8)
            ax.text(0.01, 0.974, title_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=textbox_props, zorder=2)
        elif d == 3:
            ax.set_ylabel('Feature max', fontsize=11)
            #ax.set_title(model.replace('features2-module.', ''), fontsize=9, loc='left', y=1.1, pad=-20, x = 0.01, zorder=2)
            title_text = model.replace('features2-module.', '')
            textbox_props = dict(boxstyle='round, pad=0.3', facecolor='white', edgecolor='black', linewidth=1, alpha=0.8)
            ax.text(0.01, 0.974, title_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', bbox=textbox_props, zorder=2)
        # if k == 0:
        #     leg = ax.legend(bbox_to_anchor=(1., -0.27), ncol=5,fontsize=10.5)
        #     #leg.get_texts()[-1].set_fontweight('bold')


        # axins2.set_xticks([])
        # axins2.set_yticks([])
        # axins2.tick_params(labelleft=False, labelbottom=False)
        # mark_inset(ax, axins2, loc1=3, loc2=4, fc="none", ec="0.2")

        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid()
    import matplotlib.lines as mlines

    red_dashed_line = mlines.Line2D([], [], color='red', linestyle='--', label='RMS spike in embedding layer which preceeds loss spike')

    grey_dashed_line = mlines.Line2D([], [], color='gray', linestyle='--', label='Inf gradient causing grad scaler decrease')

    # Add the custom legend to the plot
    ax.legend(handles=[red_dashed_line, grey_dashed_line], bbox_to_anchor=(0.5, -0.27), ncol=2,fontsize=10.5)


    fig.subplots_adjust(
        top=0.95, left=0.07, right=0.9, bottom=0.3, wspace=0.32, hspace=0.35
    )

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/paper3/amp.pdf', bbox_inches='tight')
