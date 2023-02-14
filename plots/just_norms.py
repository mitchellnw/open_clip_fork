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
#     'module.visual.class_embedding',
    #'module.visual.transformer.resblocks.0.attn.in_proj_weight',
    'module.visual.transformer.resblocks.0.mlp.c_fc.weight',
    'module.visual.transformer.resblocks.10.mlp.c_fc.weight',
    'module.visual.transformer.resblocks.20.mlp.c_fc.weight',
    'module.visual.transformer.resblocks.30.mlp.c_fc.weight',
    #'module.visual.transformer.resblocks.10.attn.in_proj_weight',
    # 'module.visual.transformer.resblocks.10.attn.in_proj_weight',
    # 'module.visual.transformer.resblocks.10.mlp.c_fc.weight',
    # 'module.visual.transformer.resblocks.20.attn.in_proj_weight',
    # 'module.visual.transformer.resblocks.20.mlp.c_fc.weight',
#     #'module.logit_scale',
     #'module.visual.conv1.weight',
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
    log_level =3 #+ len(modules)

    # NOTE: LOOK AT FEATURE STDDEV!
    alpha = 1
    file_list = []
    bsz = 4096*4
    file_list = [
        
        #(f'clipadamw-amp-ViTDP-H-14-16384-2e-3-0.99-v1', 'fp16-99','gray', -1),#cmap(1./3), -1),
        # (f'clipadamw-amp-ViTDP-L-14-16384-2e-3-0.95-v1', 'fp16-95',cmap(0.), -1),
        # (f'clipadamw-amp-ViTDP-L-14-16384-2e-3-0.98-v1', 'fp16-98',cmap(0.5), -1),
        # (f'clipadamw-amp-ViTDP-L-14-16384-2e-3-0.99-v1', 'fp16-99',cmap(1.), -1),

        # (f'customadamw-amp-ViT-H-14-16384-2e-3-0.99-v1', 'fp16-99',cmap(1./3), -1),
        # (f'customadamw-amp-ViT-H-14-16384-2e-3-0.98-v1', 'fp16-98',cmap(2./3), -1),
        # (f'customadamw-amp-ViT-H-14-16384-2e-3-0.95-v1', 'fp16-95',cmap(1.), -1),

       # (f'clipadamw-amp-ViT-H-14-{16384}-2e-3-0.99-v1', 'clip - batchsize 16k - l/14 0.99', 'C1', -1),

        (f'clipadamw-amp-ViTls0-H-14-{16384}-2e-3-0.98-v1', 'clip - batchsize 16k - h/14 0.98', 'gray', -1),


        #(f'customadamw-amp-ViT-H-14-{16384}-2e-3-0.98-v1', 'standard - batchsize 16k - l/14', 'C0', -1),
        #(f'customadamw-amp-ViTDP-H-14-{16384}-2e-3-0.98-v1', 'standard - batchsize 16k - l/14 + dpn', 'C1', -1),
        #(f'clipadamw-amp-ViT-H-14-{16384}-2e-3-0.98-v1', 'clip - batchsize 16k - l/14', 'C2', -1),
        #(f'clipadamw-amp-ViTDP-H-14-{16384}-2e-3-0.98-v1', 'clip - batchsize 16k - l/14 + dpn', 'C3', -1),
        #(f'customadamw-amp-ViTDP-H-14-{16384}-2e-3-0.98-gradclip2-v1', 'GC - batchsize 16k - l/14 + dpn', 'C4', -1),

        #(f'customadamw-amp-ViTDP-H-14-16384-2e-3-0.95-v1', 'dpn',cmap(1./3), -1),
        #(f'clipadamw-amp-ViT-H-14-16384-2e-3-0.95-v1', 'standard + clip',cmap(2./3), -1),
        #(f'clipadamw-amp-ViTDP-H-14-16384-2e-3-0.95-v1', 'dpn + clip',cmap(3./3), -1),

        # (f'customadamw-ViTDP-B-32-16384-2e-3-0.999-v0', 'fp16-995',cmap(0.), -1),#cmap(1./3), -1),
        # (f'customadamw-ViTDP-B-32-16384-2e-3-0.995-v0', 'fp16-995',cmap(0.), -1),#cmap(1./3), -1),
        # (f'customadamw-ViTDP-B-32-16384-2e-3-0.99-v0', 'fp16-99',cmap(1./3), -1),#cmap(1./3), -1),
        # (f'customadamw-ViTDP-B-32-16384-2e-3-0.98-v0', 'fp16-98',cmap(2./3), -1),
        # (f'customadamw-ViTDP-B-32-16384-2e-3-0.95-v0', 'fp16-95',cmap(1.), -1),

        # (f'clipadamw-amp-ViTDP-L-14-16384-2e-3-0.99-v1', 'fp16-99','gray', -1),#cmap(1./3), -1),
        # (f'clipadamw-amp-ViTDP-L-14-16384-2e-3-0.98-v1', 'fp16-98',cmap(2./3), -1),
        # (f'clipadamw-amp-ViTDP-L-14-16384-2e-3-0.95-v1', 'fp16-95',cmap(1.), -1),

        # (f'customadamw-amp-ViTDP-H-14-16384-2e-3-0.95-gradclip2-v1', 'fp16-95',cmap(0), -1),
        # (f'customadamw-amp-ViTDP-H-14-16384-2e-3-0.98-gradclip2-v1', 'fp16-98',cmap(0.5), -1),
        # (f'customadamw-amp-ViTDP-H-14-16384-2e-3-0.99-gradclip2-v1', 'fp16-99',cmap(1.), -1),

        #(f'customadamw-amp-ViT-L-14-16384-2e-3-0.98-v1-nopd', 'fp16-98', 'gray', -1),
        #(f'customadamw-amp-ViT-H-14-16384-2e-3-0.98-v1', 'fp16-98',cmap(0.), -1),
        # (f'clipadamw-amp-ViTDP-H-14-16384-2e-3-0.995-v1', 'fp16-98',cmap(1.), -1),
        # (f'clipadamw-amp-ViTDP-H-14-16384-2e-3-0.98-v1', 'fp16-98',cmap(0), -1),

        #(f'customadamw-amp-ViTDP-H-14-16384-2e-3-0.98-v1', 'fp16-98',cmap(1./3), -1),
    ]


    fig, axlist = plt.subplots(log_level, 1, figsize=(16, 5 * log_level))
    if log_level == 1:
        axlist = [axlist]

    axlabels = []

    #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
    
    for j, (file, name, color, lim) in enumerate(file_list):

        if not os.path.exists(f'/fsx/home-mitchellw/experimetns/opt/{file}/data/0/loss.csv'):
            continue

        if log_level >= 1:
            ax = axlist[0]
            for i in range(1):
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/opt/{file}/data/{i}/loss.csv', names=list(range(2)))
                df = proc(df, lim)
                #df = df[df[0] > 30000]
                #ax.set_yscale('log')
                ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), color=color, label=name)#, alpha=0.5)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
                
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                #ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=1)
                print(file)
                
                print(df.iloc[-1, 0])
                ax.set_ylabel('Loss', fontsize=16)
                #ax.set_yscale('log')
                #ax.set_xscale('log')

        # if log_level >= 2:
        #     ax = axlist[1]
        #     for i in range(1):
        #         filename = f'/fsx/home-mitchellw/experimetns/opt/{file}/data/{i}/amp.csv'
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
        #         #ax.set_yscale('log')
        #         #ax.set_xscale('log')


        for jj, module in enumerate(modules):
            ax = axlist[1]

            for i in range(1):
                #layer = 'params-module.logit_scale.csv'
                #layer = 'params-module.positional_embedding.csv' # YES!
                #layer = 'params-module.text_projection.csv' # NO!
                layer = f'params-{module}.csv'
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/opt/{file}/data/{i}/{layer}', names=list(range(17+5+4+2)))
                df = proc(df, lim)
                
                # if j == 1:
                #     alpha = 0.25
                idx = 14
                #idx=1
                idx = 6
                #idx = 18
                #if jj == 0:
                ax.plot(df.iloc[:, 0], np.sqrt(df.iloc[:, idx]),label=layer)
                ax.set_yscale('log')
                #ax.plot(df.iloc[:, 0], df.iloc[:, idx], color=color, alpha=alpha)
                #ax.set_ylabel('root(mean(square(g/u)))', fontsize=16)
                ax.set_ylabel('max gradient magnitude', fontsize=16)

                #ax.plot(df.iloc[:, 0], df.iloc[:, idx], color=color, alpha=alpha)
                #ax.plot(df.iloc[:, 0], np.sqrt(df.iloc[:, idx]), color=color, alpha=alpha)
                #ax.set_title(module, fontsize=16, y=1.0, pad=-14)
                # else:
                #     ax.plot(df.iloc[:, 0], np.sqrt(df.iloc[:, 4]), color=color, alpha=alpha)
                    
                #     ax.set_ylabel('RMS(g)', fontsize=16)
                #     ax.set_title(module, fontsize=16, y=1.0, pad=-14)


        ax = axlist[-1]
        for i in range(1):
            for layer in [
                f'features2-module.visual.transformer.resblocks.0.csv',
                f'features2-module.visual.transformer.resblocks.10.csv',
                f'features2-module.visual.transformer.resblocks.20.csv',
                f'features2-module.visual.transformer.resblocks.30.csv'
            ]:
                filename = f'/fsx/home-mitchellw/experimetns/opt/{file}/data/{i}/{layer}'
                if not os.path.exists(filename):
                    continue
                df = pd.read_csv(filename, names=list(range(17+5+4+2)))
                df = proc(df, lim)
                ax.plot(df.iloc[:, 0], df.iloc[:, 3],label=layer)
                ax.set_ylabel('feature maxs', fontsize=16)
                ax.set_yscale('log')

        # ax = axlist[-1]    
        # for i in range(1):
        #     fname = f'/fsx/home-mitchellw/experimetns/opt/{file}/checkpoints/eval.pt'
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
        # vv = 1498
        # dd = 30
        
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

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/just_norms.png', bbox_inches='tight')



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