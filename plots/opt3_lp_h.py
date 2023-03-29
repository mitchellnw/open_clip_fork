import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar

modules = [
    'module.positional_embedding',
    'module.text_projection',
    'module.logit_scale',
    'module.visual.class_embedding',
    'module.visual.positional_embedding',
    'module.visual.proj',
    'module.visual.conv1.weight',
    #'module.visual.patchnorm_pre_ln.weight',
    'module.visual.ln_pre.weight',
    'module.visual.ln_pre.bias',
    'module.visual.transformer.resblocks.0.ln_1.weight',
    'module.visual.transformer.resblocks.0.ln_1.bias',
    'module.visual.transformer.resblocks.0.attn.in_proj_weight',
    'module.visual.transformer.resblocks.0.attn.in_proj_bias',
    'module.visual.transformer.resblocks.0.attn.out_proj.weight',
    'module.visual.transformer.resblocks.0.attn.out_proj.bias',
    'module.visual.transformer.resblocks.0.ln_2.weight',
    'module.visual.transformer.resblocks.0.ln_2.bias',
    'module.visual.transformer.resblocks.0.mlp.c_fc.weight',
    'module.visual.transformer.resblocks.0.mlp.c_fc.bias',
    'module.visual.transformer.resblocks.0.mlp.c_proj.weight',
    'module.visual.transformer.resblocks.0.mlp.c_proj.bias',
    'module.visual.transformer.resblocks.10.ln_1.weight',
    'module.visual.transformer.resblocks.10.ln_1.bias',
    'module.visual.transformer.resblocks.10.attn.in_proj_weight',
    'module.visual.transformer.resblocks.10.attn.in_proj_bias',
    'module.visual.transformer.resblocks.10.attn.out_proj.weight',
    'module.visual.transformer.resblocks.10.attn.out_proj.bias',
    'module.visual.transformer.resblocks.10.ln_2.weight',
    'module.visual.transformer.resblocks.10.ln_2.bias',
    'module.visual.transformer.resblocks.10.mlp.c_fc.weight',
    'module.visual.transformer.resblocks.10.mlp.c_fc.bias',
    'module.visual.transformer.resblocks.10.mlp.c_proj.weight',
    'module.visual.transformer.resblocks.10.mlp.c_proj.bias',
    'module.visual.ln_post.weight',
    'module.visual.ln_post.bias',
    'module.token_embedding.weight',
    'module.ln_final.bias',
    'module.ln_final.weight',
]

modules = [
#'module.visual.class_embedding',
# 'module.visual.transformer.resblocks.0.attn.in_proj_weight',
#      'module.visual.transformer.resblocks.0.attn.out_proj.weight',
#      'module.visual.transformer.resblocks.0.mlp.c_proj.weight',
#      'module.visual.transformer.resblocks.0.mlp.c_proj.weight',
# #     'module.visual.transformer.resblocks.10.attn.in_proj_weight',
# #     'module.visual.transformer.resblocks.10.attn.in_proj_weight',
#      'module.visual.transformer.resblocks.10.mlp.c_fc.weight',
#      'module.visual.transformer.resblocks.20.attn.out_proj.weight',
# #     'module.visual.transformer.resblocks.20.mlp.c_fc.weight',
# #     'module.logit_scale',
# 'module.positional_embedding',
# 'module.visual.positional_embedding',
'module.visual.conv1.weight',
#'module.visual.patchnorm_pre_ln.weight',
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
    min_loss = 10
    max_scaler = 1
    log_level =2 + len(modules)

    # NOTE: LOOK AT FEATURE STDDEV!
    alpha = 1
    file_list = []
    bsz = 4096*4
    ll = -1
    file_list = [        
        # B

        # H
        # H
        (f'clipadamw-ViT-H-14-16384-2e-3-0.98-v0', 'H/14 CLIP adamw lr 2e-3, betas=(0.9, 0.98)','C0', ll), # 0.5757
        (f'clipadamw-int8-ViT-H-14-16384-2e-3-0.98-v0', 'H/14 int8 real','C1', ll),
        (f'clipadamw-int8mix-ViT-H-14-16384-2e-3-0.98-v0', 'H/14 int8 real mix','C2', ll),
        (f'clipadamw-slint8-ViT-H-14-16384-2e-3-0.98-v0', 'slint','k', ll), # 0.5757
        (f'clipadamw-sglint8-ViT-H-14-16384-2e-3-0.98-v0', 'sglint','C3', ll), # 0.5757
        (f'clipadamw-slint8v2-ViT-H-14-16384-2e-3-0.98-v0', 'slintv2','C4', ll), # 0.5757

        #(f'customadamw-ViT-H-14-16384-2e-3-0.98-v0', 'H/14 CLIP adamw lr 2e-3, betas=(0.9, 0.98)','C1', ll), # 0.5757

        # (f'clipadamw-int8-ViT-H-14-16384-2e-3-0.98-v0', 'H/14 int8 real','C1', ll),
        # (f'clipadamw-int8mix-ViT-H-14-16384-2e-3-0.98-v0', 'H/14 int8 real mix','C2', ll),
        # (f'clipadamw-camp65kfp8newsim-ViT-H-14-16384-2e-3-0.98-v0', 'fp8 sim mix row-wise/global','C4', ll),
        # (f'clipadamw-camp65kfp8global-ViT-H-14-16384-2e-3-0.98-v0', 'fp8 sim just global','C5', ll),
        # (f'clipadamw-ampfp8global-ViT-H-14-16384-2e-3-0.98-v0', 'fp8 sim just global v2','C6', ll),
        # (f'customadamw-ampfp8global-ViT-H-14-16384-2e-3-0.98-gc1-v0', 'fp8 sim just global + gc','k', ll),
        # (f'clipadamw-camp65kfp8global-ViTls0-H-14-16384-2e-3-0.98-v0', 'fp8 sim just global + ls','gray', ll),

        #(f'skipadamw-ViT-H-14-16384-2e-3-0.98-v0', 'skip v0','C4', ll),
        #(f'skipadamw-ViT-H-14-16384-2e-3-0.98-v4', 'skip v2','C6', ll),
        # (f'clipadamw-camp65kfp4sim2-ViT-H-14-16384-2e-3-0.98-v0', 'fp4 sim2','C7', ll),

        #(f'clipadamw-camp65kfp4sim4-ViT-H-14-16384-2e-3-0.98-v0', 'fp4 sim4','C7', ll),

        # (f'lion-ViT-H-14-16384-2e-4-0.98-wd2beta195-v0', 'H/14 CLIP lion lr 2e-4, betas=(0.95, 0.98), wd=2','C1', ll), # 0.5605
        # (f'lion-ViT-H-14-16384-2e-4-0.99-wd2-v0', 'H/14 CLIP lion, lr 2e-4, betas=(0.9, 0.99), wd=2','C2', ll), # 0.5488
        # (f'lion-ViT-H-14-16384-5e-4-0.98-wd1beta195-v0', 'H/14 CLIP lion, lr 5e-4, betas=(0.95, 0.98), wd=1','C3', ll),


        ## LION
        # (f'clipadamw-camp65kfp4sim3-ViT-H-14-16384-2e-3-0.98-v0', 'fp4 sim3','C8', ll),

# lion-ViT-B-32-16384-2e-4-0.99-v0
# lion-ViT-H-14-16384-2e-4-0.98-wd2beta195-v0
# lion-ViT-H-14-16384-2e-4-0.99-wd2-v0
# lion-ViT-H-14-16384-5e-4-0.98-wd1beta195-v0
# lion-ViT-H-14-16384-5e-4-0.98-wd2beta195-v0


        #(f'clipadamw-camp65kfp8-ViT-H-14-16384-2e-3-0.98-v0', 'fp8 sim','C3', ll),
        # (f'clipadamw-ViT-H-14-16384-2e-3-0.98-extraln-v0', 'update clip','C8', ll),
        # (f'clipadamw-ViTls0-H-14-16384-2e-3-0.98-v0', 'ls0','C9', ll),
        # (f'clipadamw-ViTDP-H-14-16384-2e-3-0.99-v0', 'dp','C0', ll),
        # (f'lion-ViT-H-14-16384-2e-4-0.98-wd2beta195-v0', 'lion','C4', ll),
        # (f'lion-ViT-H-14-16384-2e-4-0.99-wd2-v0', 'lion','C5', ll),
        # (f'lion-ViT-H-14-16384-5e-4-0.98-wd2beta195-v0', 'lion','C6', ll),

        

# lion-ViT-H-14-16384-2e-4-0.99-wd2-v0
# lion-ViT-H-14-16384-5e-4-0.98-wd1beta195-v0
# lion-ViT-H-14-16384-5e-4-0.98-wd2beta195-v0

        #(f'lion-ViT-H-14-16384-2e-4-0.98-wd2beta195-v0', 'lion','C4', ll),

    ]


    fig, axlist = plt.subplots(log_level, 1, figsize=(16, 5 * log_level))
    if log_level == 1:
        axlist = [axlist]

    axlabels = []

    #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
    
    for j, (file, name, color, lim) in enumerate(file_list):

        if not os.path.exists(f'/fsx/home-mitchellw/experimetns/opt3/{file}/data/0/loss.csv'):
            continue

        if log_level >= 1:
            ax = axlist[0]
            for i in range(1):
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/opt3/{file}/data/{i}/loss.csv', names=list(range(2)))
                df = proc(df, lim)
                #df = df[df[0] > 30000]
                #ax.set_yscale('log')
                #ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), color=color,alpha=1, label=name)#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
                
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=2)
                print(file)
                
                print(df.iloc[-1, 0])
                ax.set_ylabel('Loss', fontsize=16)
                #ax.set_yscale('log')
                #ax.set_xscale('log')

        if log_level >= 2:
            ax = axlist[1]
            for i in range(1):
                filename = f'/fsx/home-mitchellw/experimetns/opt3/{file}/data/{i}/amp.csv'
                if not os.path.exists(filename):
                    continue
                df = pd.read_csv(filename, names=list(range(2)))
                if len(df) == 0:
                    continue
                df = proc(df, lim)
                #df = df[df[0] > 30000]
                #ax.set_yscale('log')
                ax.plot(df.iloc[:, 0], df.iloc[:, 1], color=color, label=name)#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
                

                print(df.iloc[-1, 0])
                ax.set_ylabel('Amp', fontsize=16)
                ax.set_yscale('log')
                #ax.set_xscale('log')


        for jj, module in enumerate(modules):
            if jj + 2 >= log_level :#- 2:
                continue
            ax = axlist[jj+1]

            for i in range(1):
                #layer = 'params-module.logit_scale.csv'
                #layer = 'params-module.positional_embedding.csv' # YES!
                #layer = 'params-module.text_projection.csv' # NO!
                layer = f'params-{module}.csv'
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/opt3/{file}/data/{i}/{layer}', names=list(range(17+5+4+2)))
                df = proc(df, lim)
                
                # if j == 1:
                #     alpha = 0.25
                #idx = 14
                #idx=1
                idx = 14
                #idx = 14
                #idx = 18
                #if jj == 0:
                ax.plot(df.iloc[:, 0], np.sqrt(df.iloc[:, idx]), color=color, alpha=alpha)
                badness = 0
                # for axv in df[np.isnan(df[6])][0].values:
                #     ax.axvline(axv, color='red', alpha = 0.5)
                #     badness += 1
                # for axv in df[np.isinf(df[6])][0].values:
                #     ax.axvline(axv, color='red', linestyle='--', alpha = 0.5)
                #     badness += 1
                print('badness',module, badness)
                #ax.plot(df.iloc[:, 0], df.iloc[:, idx], color=color, alpha=alpha)
                ax.set_ylabel('grad magnitude', fontsize=16)
                #ax.set_ylabel('max gradient magnitude', fontsize=16)
                #ax.set_ylabel('mean gradient magnitude', fontsize=16)

                #ax.plot(df.iloc[:, 0], df.iloc[:, idx], color=color, alpha=alpha)
                #ax.plot(df.iloc[:, 0], np.sqrt(df.iloc[:, idx]), color=color, alpha=alpha)
                ax.set_title(module, fontsize=16, y=1.0, pad=-14)
                #ax.set_yscale('log')
                # else:
                #     ax.plot(df.iloc[:, 0], np.sqrt(df.iloc[:, 4]), color=color, alpha=alpha)
                    
                #     ax.set_ylabel('RMS(g)', fontsize=16)
                #     ax.set_title(module, fontsize=16, y=1.0, pad=-14)

        
        ax = axlist[-1]
        for i in range(1):
            layer = f'features3.0-module.visual.transformer.resblocks.30.csv'
            filename = f'/fsx/home-mitchellw/experimetns/opt3/{file}/data/{i}/{layer}'
            if not os.path.exists(filename):
                continue
            df = pd.read_csv(filename, names=list(range(17+5+4+2)))
            df = proc(df, lim)
            
            if not layer.startswith('features3.2'):
                ax.plot(df.iloc[:, 0], df.iloc[:, 2], color=color, alpha=0.5)
                ax.plot(df.iloc[:, 0], df.iloc[:, 3], color=color, alpha=alpha)
            else:
                #ax.plot(df.iloc[:, 0], 1-df.iloc[:, 2], color=color, alpha=0.5)
                ax.plot(df.iloc[:, 0], df.iloc[:, 3], color=color, alpha=0.5)
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(df.iloc[:, 3], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=2)
                #ax.set_yscale('log')

            #ax.set_ylabel('feature max to inproj, layer10', fontsize=16)

        # ax = axlist[-1]    
        # for i in range(1):
        #     fname = f'/fsx/home-mitchellw/experimetns/opt3/{file}/checkpoints/eval.pt'
        #     #beta2 = float(fname.split('-')[-2])
        #     axlabels.append(name)
        #     top1 = get_metrics(fname)
        #     if top1 > 0:
        #         ax.scatter(j, top1, color='C0')


    for j, ax in enumerate(axlist):

        ax.legend()
        #ax.set_xlim([22000, 35000])
        ax.grid()
        ax.set_xlabel('Iterations', fontsize=16)
        continue
        vv = 3180
        dd = 50
        vv = 3337
        dd = 100
        vv = 2682
        dd = 50
        vv = 2893
        dd = 40
        vv = 11800
        dd = 2e3
        vv = 2924
        dd = 25
        vv = 14677
        dd = 50
        vv = 9500
        dd = 500
        vv = 4200
        dd = 250
        vv = 2893
        dd = 50
        vv = 2600
        dd = 50
        vv = 524
        dd = 50
        
        ax.axvline(vv, linestyle='--', color='gray')
        ax.set_xlim(vv-dd, vv+dd)
        #ax.axvline(vv + 5, linestyle='--', color='gray')
        continue
        #ax.axvline(5347, linestyle='--', color='gray')

        continue
        # continue
        if j == 1:
            ax.axhline(np.sqrt(df[df[0] == vv][idx].values[-1]), color='gray', linestyle='--')
            #ax.axhline(df[df[0] == vv][idx].values[-1], color='gray', linestyle='--')
        elif j == 2:
            ax.axhline(np.sqrt(df[df[0] == vv][4].values[-1]), color='gray', linestyle='--')
        continue

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/opt3_lp_h.png', bbox_inches='tight')



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