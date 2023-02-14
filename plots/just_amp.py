import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


modules = [
    # 'module.positional_embedding',
    # 'module.text_projection',
    # 'module.logit_scale',
    # 'module.visual.class_embedding',
    # 'module.visual.positional_embedding',
    # 'module.visual.proj',
    # 'module.visual.conv1.weight',
    # 'module.visual.conv1.weight',
    # 'module.visual.conv1.weight',
    # 'module.visual.ln_pre.weight',
    # 'module.visual.ln_pre.bias',
    # 'module.visual.transformer.resblocks.0.ln_1.weight',
    # 'module.visual.transformer.resblocks.0.ln_1.bias',
    # 'module.visual.transformer.resblocks.0.attn.in_proj_weight',
    # 'module.visual.transformer.resblocks.0.attn.in_proj_bias',
    # 'module.visual.transformer.resblocks.0.attn.out_proj.weight',
    # 'module.visual.transformer.resblocks.0.attn.out_proj.bias',
    # 'module.visual.transformer.resblocks.0.ln_2.weight',
    # 'module.visual.transformer.resblocks.0.ln_2.bias',
    # 'module.visual.transformer.resblocks.0.mlp.c_fc.weight',
    # 'module.visual.transformer.resblocks.0.mlp.c_fc.bias',
    # 'module.visual.transformer.resblocks.0.mlp.c_proj.weight',
    # 'module.visual.transformer.resblocks.0.mlp.c_proj.bias',
    # 'module.visual.transformer.resblocks.10.ln_1.weight',
    # 'module.visual.transformer.resblocks.10.ln_1.bias',
    # 'module.visual.transformer.resblocks.10.attn.in_proj_weight',
    # 'module.visual.transformer.resblocks.10.attn.in_proj_bias',
    # 'module.visual.transformer.resblocks.10.attn.out_proj.weight',
    # 'module.visual.transformer.resblocks.10.attn.out_proj.bias',
    # 'module.visual.transformer.resblocks.10.ln_2.weight',
    # 'module.visual.transformer.resblocks.10.ln_2.bias',
    # 'module.visual.transformer.resblocks.10.mlp.c_fc.weight',
    # 'module.visual.transformer.resblocks.10.mlp.c_fc.bias',
    # 'module.visual.transformer.resblocks.10.mlp.c_proj.weight',
    # 'module.visual.transformer.resblocks.10.mlp.c_proj.bias',
    # 'module.visual.ln_post.weight',
    # 'module.visual.ln_post.bias',
    # 'module.token_embedding.weight',
    # 'module.ln_final.bias',
    # 'module.ln_final.weight',
]
cmap=plt.get_cmap('cool')
def get_metrics(filename):
    if not os.path.exists(filename):
        return -1
    f = open(filename, "r")
    c = f.read()
    good =[l for l in c.split("\n") if "imagenet-zero" in l]
    if len(good) != 1:
        return -1
    top5 = float(good[0].split("\t")[1].split(" ")[-1])
    top1 = float(good[0].split("\t")[0].split(" ")[-1])
    return top1

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

    file_list = []
    bsz = 4096*4
    file_list = [
        (f'customadamw-ViT-B-32-{bsz}-2e-3-0.8-v0', 0.8, cmap(1.), -1),
        (f'customadamw-ViT-B-32-{bsz}-2e-3-0.9-v0', 0.9, cmap(0.5), -1),
        (f'customadamw-ViT-B-32-{bsz}-2e-3-0.99-v0', 0.99, cmap(1.), -1),
    ]

    fig, axlist = plt.subplots(log_level, 1, figsize=(16, 5 * log_level))
    if log_level == 1:
        axlist = [axlist]

    axlabels = []

    #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
    ax = axlist[-1]    

    for template, name, color in [

        (f'customadamw-amp-ViT-H-14-{16384}-2e-3-' + '{}-v1', 'standard - batchsize 16k - l/14', 'C0'),
        (f'customadamw-amp-ViTDP-H-14-{16384}-2e-3-' + '{}-v1', 'standard - batchsize 16k - l/14 + dpn', 'C1'),
        (f'clipadamw-amp-ViT-H-14-{16384}-2e-3-' + '{}-v1', 'clip - batchsize 16k - l/14', 'C2'),
        (f'clipadamw-amp-ViTDP-H-14-{16384}-2e-3-' + '{}-v1', 'clip - batchsize 16k - l/14 + dpn', 'C3'),


        # (f'customadamw-ViT-H-14-{16384}-2e-3-' + '{}-v1', 'standard - batchsize 16k - l/14', 'C0'),
        # (f'customadamw-ViTDP-H-14-{16384}-2e-3-' + '{}-v1', 'standard - batchsize 16k - l/14 + dpn', 'C1'),
        # (f'clipadamw-ViT-H-14-{16384}-2e-3-' + '{}-v1', 'clip - batchsize 16k - l/14', 'C2'),
        # (f'clipadamw-ViTDP-H-14-{16384}-2e-3-' + '{}-v1', 'clip - batchsize 16k - l/14 + dpn', 'C3'),


        # (f'customadamw-amp-ViT-L-14-{16384}-2e-3-' + '{}-v1', 'standard - batchsize 16k - l/14', 'C0'),
        # (f'customadamw-amp-ViTDP-L-14-{16384}-2e-3-' + '{}-v1', 'standard - batchsize 16k - l/14 + dpn', 'C1'),
        # (f'clipadamw-amp-ViT-L-14-{16384}-2e-3-' + '{}-v1', 'clip - batchsize 16k - l/14', 'C2'),
        # (f'clipadamw-amp-ViTDP-L-14-{16384}-2e-3-' + '{}-v1', 'clip - batchsize 16k - l/14 + dpn', 'C3'),


        # (f'customadamw-ViT-L-14-{16384}-2e-3-' + '{}-v1', 'standard - batchsize 16k - l/14', 'C0'),
        # (f'customadamw-ViTDP-L-14-{16384}-2e-3-' + '{}-v1', 'standard - batchsize 16k - l/14 + dpn', 'C1'),
        # (f'clipadamw-ViT-L-14-{16384}-2e-3-' + '{}-v1', 'clip - batchsize 16k - l/14', 'C2'),
        # (f'clipadamw-ViTDP-L-14-{16384}-2e-3-' + '{}-v1', 'clip - batchsize 16k - l/14 + dpn', 'C3'),


        # (f'customadamw-ViT-B-32-{16384}-2e-3-' + '{}-v0', 'standard - batchsize 16k - b/32', 'C0'),
        # (f'customadamw-ViTDP-B-32-{16384}-2e-3-' + '{}-v0', 'standard - batchsize 16k - b/32 + dpn', 'C1'),
        # (f'clipadamw-ViT-B-32-{16384}-2e-3-' + '{}-v0', 'clip - batchsize 16k - b/32', 'C2'),
        # (f'clipadamw-ViTDP-B-32-{16384}-2e-3-' + '{}-v0', 'clip - batchsize 16k - b/32 + dpn', 'C3'),

        # (f'customadamw-ViT-B-32-{16384}-1e-3-' + '{}-v0', 'standard - batchsize 16k - lr 1e-3', 'C1'),
        # (f'customadamw-ViT-B-32-{16384}-5e-4-' + '{}-v0', 'standard - batchsize 16k - lr 5e-4', 'C2'),
        #(f'customadamw-ViT-B-32-{16384*4*4}-2e-3-' + '{}-v0', 'standard - batchsize 64k', 'C1')
        #(f'clipadamw-ViT-B-32-{16384}-2e-3-' + '{}-v0', 'clipping - batchsize 64k', 'C2'),
        #(f'customadamw-ViTDP-B-32-{16384*4}-2e-3-' + '{}-v0', 'patchnorm - batchsize 64k', 'C3'),
        # (f'customadamw-0.95-ViT-B-32-{16384}-2e-3-' + '{}-v0', 'momentum 0.95 - batchsize 16k', 'C4'),
        # (f'customadamw-0.98-ViT-B-32-{16384}-2e-3-' + '{}-v0', 'momentum 0.98 - batchsize 16k', 'C5'),
        # (f'customadamw-0.99-ViT-B-32-{16384}-2e-3-' + '{}-v0', 'momentum 0.99 - batchsize 16k', 'C6'),
        #(f'customadamw0.99-ViT-B-32-{16384}-2e-3-' + '{}-v0', 'standard - batchsize 16k', 'C7'),
        #(f'customadamw-gradclip2-ViT-B-32-16384-2e-3-' + '{}-v0', 'gc - batchsize 16k', 'C1'),
    ]:
            
        xs, ys, ysmin = [], [], []
        allb2s = [0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
        for j, beta2 in enumerate(allb2s):
            fname = f'/fsx/home-mitchellw/experimetns/opt/{template.format(beta2)}/data/0/amp.csv'

            if not os.path.exists(fname):
                continue
            df = pd.read_csv(fname, names=list(range(2)))
            if len(df) == 0:
                continue
            top1 = df[1].values[-1]
            top1min = df[1].min()

            if top1 > 0:
                xs.append(j)
                ys.append(top1)
                ysmin.append(top1min)
        
        # mean_ys = np.mean(ys)
        # ax.plot(xs, [y - mean_ys for y in ys], marker='o', color=color, label=name)
        ax.plot(xs, ys, marker='o', color=color, label=name)
        #ax.plot(xs, ysmin, marker='o', color=color, label=name, linestyle='--')

    ax.set_xticks([j for j, _ in enumerate(allb2s)])
    ax.set_xticklabels(allb2s)
    ax.set_xlabel('Beta2', fontsize=16)
    ax.set_ylabel('AMP Scaler', fontsize=16)
    ax.legend()
    ax.grid()


    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/just_amp.png', bbox_inches='tight')



"""

0    stepd,
1    p.pow(2).mean().item(), #'weight_means'
2    p.pow(2).std().item(), # 'weight_std'
3    p.pow(2).max().item(), # 'weight_max'
4    p.grad.pow(2).mean().item(), #'grad sq mean'
5    p.grad.pow(2).std().item(), # 'grad sq std'
6    p.grad.pow(2).max().item(), # 'grad sq max'
7    optimizer.state[p]['exp_avg'].pow(2).mean().item(), #'v_means'
8    optimizer.state[p]['exp_avg'].pow(2).std().item(), # 'v_std'
9    optimizer.state[p]['exp_avg'].pow(2).max().item(), # 'v_max'
10    optimizer.state[p]['exp_avg_sq'].mean().item(), #'u_means'
11    optimizer.state[p]['exp_avg_sq'].std().item(), # 'u_std'
12    optimizer.state[p]['exp_avg_sq'].max().item(), # 'u_max'
13    optimizer.state[p]['exp_avg_sq'].min().item(), # 'u_min
14    optimizer.state[p]['rms_mean'],
15    optimizer.state[p]['rms_std'],
16    optimizer.state[p]['rms_min'],
17    optimizer.state[p]['rms_max'],
18    optimizer.state[p]['rms_sq_d1'],
19    optimizer.state[p]['rms_d1'],
20    optimizer.state[p]['numel'],
21    optimizer.state[p]['relu'],
22    optimizer.state[p]['beta2hat'],
"""