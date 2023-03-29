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
#     'module.visual.transformer.resblocks.0.mlp.c_fc.weight',
#     'module.visual.transformer.resblocks.0.mlp.c_proj.weight',
#     'module.visual.transformer.resblocks.10.attn.in_proj_weight',
#     'module.visual.transformer.resblocks.10.attn.in_proj_weight',
#     'module.visual.transformer.resblocks.10.mlp.c_fc.weight',
#     'module.visual.transformer.resblocks.20.attn.in_proj_weight',
#     'module.visual.transformer.resblocks.20.mlp.c_fc.weight',
#     'module.logit_scale',
#'module.positional_embedding',
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

# (f'clipadamw-int4sim2-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int4 sim2','C6', 1000),
# (f'clipadamw-int4sim3-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int4 sim3','k', 1000),

# (f'clipadamw-int4sim6-ViT-B-32-16384-2e-3-0.98-v0', 'sim block backward','C0', 1000),
# (f'clipadamw-int4sim7-ViT-B-32-16384-2e-3-0.98-v0', 'sim vector backward','C1', 1000),
if __name__ == '__main__':
    kernel_size = 40
    min_loss = 14
    max_scaler = 1
    log_level =3 + len(modules)

    # NOTE: LOOK AT FEATURE STDDEV!
    alpha = 1
    file_list = []
    bsz = 4096*4
    ll = -1
    file_list = [        
        # B
        #(f'customadamw-ViT-B-32-8192-2e-3-0.995-rmscheck-v0', 'adamw','C0', ll),
        #(f'customadamw-ViT-B-32-8192-2e-3-0.995-rmscheck-v1', 'adamw','C0', ll),
        (f'clipadamw-ViT-B-32-16384-2e-3-0.98-v0', 'adamw','C0', ll),

        (f'clipadamw-int8-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int8 real','C1', ll),
        (f'clipadamw-int8mix-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int8 real mix','C2', ll),
        (f'clipadamw-slint8-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 triton','C3', ll),


        # (f'clipadamw-camp65kfp8newsim-ViT-B-32-16384-2e-3-0.98-v0', 'fp8 sim mix row-wise/global','C4', ll),
        # (f'clipadamw-camp65kfp8global-ViT-B-32-16384-2e-3-0.98-v0', 'fp8 sim just global','C5', ll),
        # (f'clipadamw-ampfp8global-ViT-B-32-16384-2e-3-0.98-v0', 'fp8 sim just global v2','C6', ll),

        # (f'clipadamw-camp2-ViT-B-32-16384-2e-3-0.98-v0', 'badamp','k', ll),

        # (f'dog-ViT-B-32-16384-1-0.98-skipcwd1en4-v0', 'DoG (wd 1e-4)','C1', ll),
        #(f'ldog-ViT-B-32-16384-1-0.98-skipcwd1en4-v0', 'LDoG (wd 1e-4)','C2', ll),
        # (f'dog-ViT-B-32-16384-1-0.98-skipcwd1en4v2-v0', 'DoG (wd 1e-4, reps_rel=1e-4)','C3', ll),
        # (f'ldog-ViT-B-32-16384-1-0.98-skipcwd1en4v2-v0', 'LDoG (wd 1e-4, reps_rel=1e-4)','C4', ll),
        #(f'customadamw-ViT-B-32-16384-2e-3-0.98-gc1-v0', 'adamw + gc','C5', ll),


        
        #(f'customadamw-ViT-B-32-16384-2e-3-0.98-v0', 'raw-adamw','C1', -1),

        #(f'clipadamw-int8-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int8real','C1', ll),
        # (f'clipadamw-int8mix-ViT-B-32-16384-2e-3-0.98-v0', 'In8 mix','C2', -1),
        # (f'clipadamw-camp65kfp8newsim-ViT-B-32-16384-2e-3-0.98-v0', 'FP8 newsim','C3', -1),
        # (f'clipadamw-camp65kint8newsim-ViT-B-32-16384-2e-3-0.98-v0', 'In8 newsim','C4', -1),
\
# clipadamw-camp65kint4sim1-ViT-B-32-16384-2e-3-0.98-v0
# clipadamw-camp65kint4sim2-ViT-B-32-16384-2e-3-0.98-v0
        # (f'clipadamw-camp65kint4sim2-ViT-B-32-16384-2e-3-0.98-v0', 'In4 Sim New','C3', 1000),


        # (f'clipadamw-int8sim-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int8','C1', 2000),
        # (f'clipadamw-int4sim-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int4 sim block fwd and backward (1024)','C2', 2000),

        # (f'clipadamw-int4sim9-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int4 sim vector fwd and backward','C3', 2000),
        # (f'clipadamw-int4sim11-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int4 sim block fwd and backward (512)','C4', 2000),

        #(f'clipadamw-int8nohalf-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int8 no half cast','C4', 500),
        #(f'clipadamw-int4sim-ViT-B-32-16384-2e-3-0.98-v0', 'B/32 int4 sim','C5', 500),

# lion-int8-ViT-B-32-16384-2e-4-0.99-v0
# clipadamw-int4sim-ViT-B-32-16384-2e-3-0.98-v0
        

        # (f'lion-ViT-B-32-16384-2e-4-0.99-v0', 'lion','C5', -1),
        # (f'ulion-ViT-B-32-16384-2e-4-0.99-v0', 'unsigned lion','C6', -1),

        # (f'lion-int8-ViT-B-32-16384-2e-4-0.99-v0', 'B/32','C6', -1),

        #(f'clipadamw-ViT-B-32-16384-2e-3-0.98-extraln-v0', 'B/32 extraln','C2', -1),
        #(f'clipadamw-ViT-B-32-16384-2e-3-0.98-extraln-v0', 'B/32 extraln','C1', -1),


        # H
        # (f'clipadamw-ViT-H-14-16384-2e-3-0.98-v0', 'H/14','C4', -1),
        # (f'clipadamw-int8-ViT-H-14-16384-2e-3-0.98-v0', 'H/14 int8','C5', -1),

        
        # (f'clipadamw-ViTls0-H-14-16384-2e-3-0.98-v0', 'H/14 ls0','C4', -1),

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

        # if log_level >= 1:
        #     ax = axlist[1]
        #     for i in range(1):
        #         filename = f'/fsx/home-mitchellw/experimetns/opt3/{file}/data/{i}/amp.csv'
        #         if not os.path.exists(filename):
        #             continue
        #         df = pd.read_csv(filename, names=list(range(2)))
        #         if len(df) == 0:
        #             continue
        #         df = proc(df, lim)
        #         #df = df[df[0] > 30000]
        #         #ax.set_yscale('log')
        #         ax.plot(df.iloc[:, 0], df.iloc[:, 1], color=color, label=name)#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
                

        #         print(df.iloc[-1, 0])
        #         ax.set_ylabel('Amp', fontsize=16)
        #         ax.set_yscale('log')
        #         #ax.set_xscale('log')


        for jj, module in enumerate(modules):
            if jj >= log_level :
                print(jj)
                continue
            ax = axlist[jj+1]

            for i in range(1):
                #layer = 'params-module.logit_scale.csv'
                #layer = 'params-module.positional_embedding.csv' # YES!
                #layer = 'params-module.text_projection.csv' # NO!
                layer = f'params-{module}.csv'
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/opt3/{file}/data/{i}/{layer}', names=list(range(17+5+4+2)))
                df = proc(df, lim)
                print(df)
                
                # if j == 1:
                #     alpha = 0.25
                #idx = 14
                #idx=1
                idx = 17
                idx = 14
                #idx = 14
                #idx = 18
                #if jj == 0:
                ax.plot(df.iloc[:, 0], np.sqrt(df.iloc[:, idx]), color=color, alpha=alpha)
                badness = 0
                for axv in df[np.isnan(df[6])][0].values:
                    ax.axvline(axv, color='red')
                    badness += 1
                for axv in df[np.isinf(df[6])][0].values:
                    ax.axvline(axv, color='red', linestyle='--')
                    badness += 1
                print('badness',module, badness)
                #ax.plot(df.iloc[:, 0], df.iloc[:, idx], color=color, alpha=alpha)
                ax.set_ylabel('root(mean(square(g/u)))', fontsize=16)
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


        # ax = axlist[-1]
        # for i in range(1):
        #     layer = f'features3.2-module.visual.transformer.resblocks.0.csv'
        #     filename = f'/fsx/home-mitchellw/experimetns/opt3/{file}/data/{i}/{layer}'
        #     if not os.path.exists(filename):
        #         continue
        #     df = pd.read_csv(filename, names=list(range(17+5+4+2)))
        #     df = proc(df, lim)
            
        #     if not layer.startswith('features3.2'):
        #         ax.plot(df.iloc[:, 0], df.iloc[:, 2], color=color, alpha=0.5)
        #         ax.plot(df.iloc[:, 0], df.iloc[:, 3], color=color, alpha=alpha)
        #     else:
        #         #ax.plot(df.iloc[:, 0], 1-df.iloc[:, 2], color=color, alpha=0.5)
        #         ax.plot(df.iloc[:, 0], 1-df.iloc[:, 3], color=color, alpha=0.5)
        #         kernel = np.ones(kernel_size) / kernel_size
        #         data_convolved = np.convolve(1-df.iloc[:, 3], kernel, mode='same')
        #         data_convolved = data_convolved[kernel_size:-kernel_size]
        #         ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=2)
        #         #ax.set_yscale('log')

        #     ax.set_ylabel('feature max to inproj, layer10', fontsize=16)

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
        vv = 3146
        dd = 50
        vv = 2470
        dd = 50
        vv = 4659
        dd = 50
        vv = 4443
        dd = 50
        dd = 500

        ax.axvline(vv, linestyle='--', color='gray')
        
        # ax.axvline(2452, linestyle='--', color='gray')
        # ax.axvline(2453, linestyle='--', color='gray')

        # ax.axvline(2471, linestyle='--', color='gray')
        # ax.axvline(2472, linestyle='--', color='gray')
        # ax.axvline(2474, linestyle='--', color='gray')
        # ax.axvline(2476, linestyle='--', color='gray')


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

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/opt3_lp.png', bbox_inches='tight')



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


