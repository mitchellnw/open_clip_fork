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
    'module.visual.conv1.weight',
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
    # 'module.visual.transformer.resblocks.11.ln_1.weight',
    # 'module.visual.transformer.resblocks.11.ln_1.bias',
    # 'module.visual.transformer.resblocks.11.attn.in_proj_weight',
    # 'module.visual.transformer.resblocks.11.attn.in_proj_bias',
    # 'module.visual.transformer.resblocks.11.attn.out_proj.weight',
    # 'module.visual.transformer.resblocks.11.attn.out_proj.bias',
    # 'module.visual.transformer.resblocks.11.ln_2.weight',
    # 'module.visual.transformer.resblocks.11.ln_2.bias',
    # 'module.visual.transformer.resblocks.11.mlp.c_fc.weight',
    # 'module.visual.transformer.resblocks.11.mlp.c_fc.bias',
    # 'module.visual.transformer.resblocks.11.mlp.c_proj.weight',
    # 'module.visual.transformer.resblocks.11.mlp.c_proj.bias',
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
    log_level = 1  + len(modules)

    # NOTE: LOOK AT FEATURE STDDEV!

    file_list = []
    file_list = [
        #('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-098-amp_bfloat16-v1', 'standard (lr=1e-3, beta2=0.98, warmup=8k, bs=32k)', 'C0', -1),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-098-amp_bfloat16-pinit-v1', 'p-init (lr=1e-3, beta2=0.98, warmup=8k, bs=32k)', 'C1', 5000),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-098-amp_bfloat16-cinit-v1', 'c-init (lr=1e-3, beta2=0.98, warmup=8k, bs=32k)', 'C3', -1),#3900),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-098-amp_bfloat16-pinit-cinit-v1', 'c-init + p-init (lr=1e-3, beta2=0.98, warmup=8k, bs=32k)', 'C2', -1),#3900),
        #('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-pinit-v1', 'p-init (lr=1e-3, beta2=0.95, warmup=8k, bs=32k)', 'C4', -1),
        #('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-v1', 'standard (lr=1e-3, beta2=0.95, warmup=8k, bs=32k)', 'C5', -1),
        #('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-0999-amp_bfloat16-v1', 'standard (lr=1e-3, beta2=0.999, warmup=8k, bs=32k)', 'C6', -1),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-0999-amp_bfloat16-v1', 'standard (lr=1e-3, beta2=0.999, warmup=8k, bs=32k)', 'C7', -1),
        #('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-05-amp_bfloat16-v1', 'standard (lr=1e-3, beta2=0.5, warmup=8k, bs=32k)', 'C8', -1),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-0999-amp_bfloat16-resume0seed0', 'resume0seed0', 'C8', -1),
        # ('clip-H-14-pd05-bs32k-w8k-opt1e-3-09-0999-amp_bfloat16-resume0seed1', 'resume0seed1', 'C9', -1),

        
        
        # ('cat-ViT-B-16-pd05-16384-2e-3-0.95-v0', 'baseline', 'C0', -1),
        # ('sepcat-ViT-B-16-pd05-16384-2e-3-0.95-v0', 'seperate QKV', 'C1', -1),
        # ('gadamwcat-ViT-B-16-pd05-16384-2e-3-0.95-v0', 'GAdamW', 'C2', -1),
        # ('gadamwdecaycat-ViT-B-16-pd05-16384-1e-3-0.95-v0', 'GAdamW\n(2x decay, half LR)', 'C3', -1),
        # ('cadamwcat-ViT-B-16-pd05-16384-2e-3-0.95-v0', 'AdamWR', 'C4', -1),

        # (f'cadamwrscat-ViT-B-16-pd05-{4096*4}-2e-3-0.99-v0', 'CAdamWRS 0.99', cmap(1.), -1),
        # (f'cadamwrscat-ViT-B-16-pd05-{4096*4}-2e-3-0.98-v0', 'CAdamWRS 0.98', cmap(0.75), -1),
        # (f'cadamwrscat-ViT-B-16-pd05-{4096*4}-2e-3-0.95-v0', 'CAdamWRS 0.95', cmap(0.5), -1),
        # (f'cadamwrscat-ViT-B-16-pd05-{4096*4}-2e-3-0.9-v0', 'CAdamWRS 0.9', cmap(0.25), -1),
        # (f'cadamwrscat-ViT-B-16-pd05-{4096*4}-2e-3-0.8-v0', 'CAdamWRS 0.8', cmap(0.), -1),


        #(f'cadamw5cat-ViT-B-16-pd05-{4096*4}-2e-3-0.99-v0', 'CAdamW 0.99', cmap(1.), -1),
        #(f'cadamw5cat-ViT-B-16-pd05-{4096*4}-2e-3-0.98-v0', 'CAdamW 0.98', cmap(0.75), -1),
        (f'cadamw4cat-ViT-B-16-pd05-{4096*4}-2e-3-0.99-v0', 'CAdamW 0.99', cmap(0.5), -1),
        #(f'cadamw4cat-ViT-B-16-pd05-{4096*4}-2e-3-0.9-v0', 'CAdamW 0.9', cmap(0.25), -1),
        #(f'cadamw5cat-ViT-B-16-pd05-{4096*4}-2e-3-0.8-v0', 'CAdamW 0.8', cmap(0.), -1),
        #('cadamw4cat-ViT-B-16-pd05-65536-2e-3-0.9-v0', 'CAdamW 0.9', cmap(0.25), -1),
        #('cadamw4cat-ViT-B-16-pd05-16384-2e-3-0.8-v0', 'CAdamW 0.8', cmap(0.), -1),



        #('cat-ViT-B-16-pd05-131072-1e-3-0.5-v0', '5', cmap(0.), -1),
        #('cat-ViT-B-16-pd05-131072-1e-3-0.8-v0', '8', cmap(0.1), -1),
        # ('cat-ViT-B-16-pd05-131072-1e-3-0.9-v0', '9', cmap(0.25), -1),
        # ('cat-ViT-B-16-pd05-131072-1e-3-0.95-v0', '95', cmap(0.5), -1),
        # ('cat-ViT-B-16-pd05-131072-1e-3-0.98-v0', '98', cmap(0.75), -1),
        # ('cat-ViT-B-16-pd05-131072-1e-3-0.99-v0', '99', cmap(1.), -1),
        
        # ('cat-ViT-B-16-pd05-65536-1e-3-0.3-ps-cap0.999-v0', 's-0.3', 'C0', -1),
        # ('cat-ViT-B-16-pd05-65536-1e-3-0.4-ps-cap0.999-v0', 's-0.4', 'C1', -1),
        # ('cat-ViT-B-16-pd05-65536-1e-3-0.5-ps-cap0.999-v0', 's-0.5', 'C2', -1),

        # ('cat-ViT-B-16-pd05-16384-1e-3-0.99-v0', '99', cmap(1.), -1),
        # ('cat-ViT-B-16-pd05-16384-1e-3-0.98-v0', '98', cmap(0.75), -1),
        # ('cat-ViT-B-16-pd05-16384-1e-3-0.95-v0', '95', cmap(0.5), -1),
        # ('cat-ViT-B-16-pd05-16384-1e-3-0.9-v0', '9', cmap(0.25), -1),
        # ('cat-ViT-B-16-pd05-16384-1e-3-0.8-v0', '8', cmap(0.), -1),
        
        # ('cat-ViT-B-16-pd05-4096-5e-4-0.99-v0', '99', cmap(1.), -1),
        # ('cat-ViT-B-16-pd05-4096-5e-4-0.98-v0', '98', cmap(0.75), -1),
        # ('cat-ViT-B-16-pd05-4096-5e-4-0.95-v0', '95', cmap(0.5), -1),
        # ('cat-ViT-B-16-pd05-4096-5e-4-0.9-v0', '9', cmap(0.25), -1),
        # ('cat-ViT-B-16-pd05-4096-5e-4-0.8-v0', '8', cmap(0.), -1),

        # ('cat-ViT-B-16-pd05-16384-4e-3-0.99-v0', '99', cmap(1.), -1),
        # ('cat-ViT-B-16-pd05-16384-4e-3-0.98-v0', '98', cmap(0.75), -1),
        # ('cat-ViT-B-16-pd05-16384-4e-3-0.95-v0', '95', cmap(0.5), -1),
        # ('cat-ViT-B-16-pd05-16384-4e-3-0.9-v0', '9', cmap(0.25), -1),
        # ('cat-ViT-B-16-pd05-16384-4e-3-0.8-v0', '8', cmap(0.), -1),

        # ('cat-ViT-H-14-pd05-65536-1e-3-0.8-ps-cap0.999-v0', '8', cmap(1.), -1),
        # ('cat-ViT-H-14-pd05-65536-1e-3-0.5-ps-cap0.999-v0', '5', cmap(0.75), -1),
        # ('cat-ViT-H-14-pd05-65536-1e-3-0.3-ps-cap0.999-v0', '3', cmap(0.5), -1),
        #('cat-ViT-B-16-pd05-65536-1e-3-0.-ls-cap0.999-v0', 'ls', cmap(0.25), -1),

        # ('cat-ViT-S-16-pd05-65536-1e-3-0.99-v0', 'Small', cmap(0.), -1),
        # ('cat-ViT-B-16-pd05-65536-1e-3-0.99-v0', 'Base', cmap(0.25), -1),
        # ('cat-ViT-L-16-pd05-65536-1e-3-0.99-v0', 'Large', cmap(0.5), -1),
        # ('cat-ViT-H-14-pd05-65536-1e-3-0.99-v0', 'Huge', cmap(0.75), -1),
    ]

    # H-14 wtach done (but pls confirm)
    # L-16 watch (but pls confirm)
    # B-16 done.
    # B-16,16k, done.
    # B-16,16k, 5e-4,not done.
    # B-16,16k, 2e-3, done.
    # B-16, 4k, done.
    # S-16 done.

    # PS
    # B-16 done.
    # L-16 done
    # H-14 .8



    #file_list = file_list[1:2]
    #file_list = file_list[:2]

    fig, axlist = plt.subplots(log_level, 1, figsize=(16, 5 * log_level))
    if log_level == 1:
        axlist = [axlist]

    axlabels = []

    #axins2 = zoomed_inset_axes(axlist[0], zoom=3, bbox_to_anchor=(1, 1))
    
    for j, (file, name, color, lim) in enumerate(file_list):

        if log_level >= 1:
            ax = axlist[0]
            for i in range(1):
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/b2cpbackup/{file}/data/{i}/loss.csv', names=list(range(2)))
                df = proc(df, lim)
                #df = df[df[0] > 30000]
                #ax.set_yscale('log')
                ax.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), color=color)#, label='beta2 = 0.99' if j ==0 else 'beta2 = 0.9')#, alpha=0.3)#, label=name)# alpha=0.5,
                
                kernel = np.ones(kernel_size) / kernel_size
                data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
                data_convolved = data_convolved[kernel_size:-kernel_size]
                #ax.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=color, label=name, linewidth=1)
                print(file)
                
                print(df.iloc[-1, 0])
                ax.set_ylabel('Loss', fontsize=16)
                #ax.set_yscale('log')
                #ax.set_xscale('log')


        for jj, module in enumerate(modules):
            if jj + 1 >= log_level :
                continue
            ax = axlist[jj+1]
            for i in range(1):
                #layer = 'params-module.logit_scale.csv'
                #layer = 'params-module.positional_embedding.csv' # YES!
                #layer = 'params-module.text_projection.csv' # NO!
                layer = f'params-{module}.csv'
                df = pd.read_csv(f'/fsx/home-mitchellw/experimetns/b2cpbackup/{file}/data/{i}/{layer}', names=list(range(17+5+4)))
                df = proc(df, lim)

                #import pdb; pdb.set_trace()
                #import pdb; pdb.set_trace()
                #ax.plot(df.iloc[:, 0], df.iloc[:, 1], color=color)
                above1 = (np.sqrt(df.iloc[:, 18]) - 1.)**2
                ax.plot(df.iloc[:, 0], np.sqrt(df.iloc[:, 18]), color=color)#, label=np.mean(above1))
                #ax.plot(df.iloc[:, 0], df.iloc[:, 23], color=color)
                
                
            
                #ax.plot(df.iloc[:, 0], df.iloc[:, 18], color=color)
                #ax.plot(df.iloc[:, 0], df.iloc[:, 5], color=color, alpha=0.6)
                #ax.plot(df.iloc[:, 0], df.iloc[:, 6], color=color, alpha=0.5)
                #ax.set_yscale('log')
                ax.set_ylabel('RMS', fontsize=16)
                ax.set_title(module, fontsize=16, y=1.0, pad=-14)


        # ax = axlist[-1]    
        # for i in range(1):
        #     fname = f'/fsx/home-mitchellw/experimetns/b2cpbackup/{file}/checkpoints/eval.pt'
        #     #beta2 = float(fname.split('-')[-2])
        #     axlabels.append(name)
        #     top1 = get_metrics(fname)
        #     if top1 > 0:
        #         ax.scatter(j, top1, color='C0')


    for j, ax in enumerate(axlist):
        # if j == log_level - 1:
        #     #ax.set_xlabel('Beta 2')
        #     #import pdb; pdb.set_trace()
        #     ax.set_xticks([jj for jj in range(len(file_list))])
        #     ax.set_xticklabels(axlabels)
        #     ax.set_ylabel('Accuracy')
        # if j == 1:
        #     #ax.set_xlabel('Beta 2')
        #     #import pdb; pdb.set_trace()
        #     ax.set_xticks([jj for jj in range(len(file_list))])
        #     ax.set_xticklabels(axlabels)
        #     ax.set_ylabel('Accuracy')
        #     #ax.set_xscale('log')
        #     ax.grid()
        # else:
        #if j == 0:
        ax.legend()
        #ax.set_xlim([22000, 35000])
        ax.grid()
        ax.set_xlabel('Iterations', fontsize=16)
        vv = 12473
        dd = 3e1
        vv = 9883
        dd = 1e1
        vv = 200
        dd = 200
        vv = 3710
        dd = 2e3
        continue
        # if j > 0:
        #     ax.set_ylim([0, 4])
        #     ax.set_xlim([0000, 1000])
        #     ax.axhline(np.sqrt(2)-1, color='gray')
        #continue
        #ax.axvline(vv, linestyle='--', color='gray')
        #ax.set_xlim(vv-dd, vv+dd)

        # ax.axvline(3687, linestyle='--', color='gray')
        # ax.axvline(3725, linestyle='--', color='gray')
        ax.set_xlim([2800, 4600])
        # ax.set_xlim([2960 - 50, 2960 + 50])
        # ax.set_xlim([3573 - 50, 3573 + 50])
        ax.set_xlim([4468 - 50, 4468 + 50])
        

        # if j == 1:
        #     ax.axhline(3)
        ax.axvline(3573, linestyle='--', color='gray')
        ax.axvline(4468, linestyle='--', color='gray')
        ax.axvline(2960, linestyle='--', color='gray')

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/b2_faws.png', bbox_inches='tight')



"""
step,
p.pow(2).sum().pow(0.5).item(), #'weight_norms'
p.abs().max().item(), # 'weight_maxs'
p.grad.pow(2).sum().pow(0.5).item(), # 'grad_norms'
p.grad.abs().max().item(), # 'grad_maxs'
optimizer.state[p]['exp_avg'].pow(2).sum().pow(0.5).item(), # 'exp_avgs_norms'
optimizer.state[p]['exp_avg'].abs().max().item(), # 'exp_avgs_maxs'
optimizer.state[p]['exp_avg_sq'].pow(2).sum().pow(0.5).item(), # 'exp_avg_sqs_norms'
optimizer.state[p]['exp_avg_sq'].abs().max().item(), # 'exp_avg_sqs_maxs'
"""

"""
0 step,
1 p.abs().mean().item(), #'weight_means'
2 p.abs().std().item(), # 'weight_std'
3 p.abs().max().item(), # 'weight_max'
4 p.grad.abs().mean().item(), #'grad_means'
5 p.grad.abs().std().item(), # 'grad_std'
6 p.grad.abs().max().item(), # 'grad_max'
7 optimizer.state[p]['exp_avg'].abs().mean().item(), #'v_means'
8 optimizer.state[p]['exp_avg'].abs().std().item(), # 'v_std'
9 optimizer.state[p]['exp_avg'].abs().max().item(), # 'v_max'
10 optimizer.state[p]['exp_avg_sq'].abs().mean().item(), #'g_means'
11 optimizer.state[p]['exp_avg_sq'].abs().std().item(), # 'g_std'
12 optimizer.state[p]['exp_avg_sq'].abs().max().item(), # 'g_max'
13 optimizer.state[p]['exp_avg_sq'].min().item(), # u_min
14 optimizer.state[p]['g2_mean'], #
15 optimizer.state[p]['g2_std'],
16 optimizer.state[p]['g2_min'],
17 optimizer.state[p]['g2_max'],
18 optimizer.state[p]['rms_mean'],
19 optimizer.state[p]['rms_std'],
20 optimizer.state[p]['rms_min'],
21 optimizer.state[p]['rms_max'],
22                                optimizer.state[p]['rms_sq_d1'],
23                                optimizer.state[p]['rms_d1'],
24                                optimizer.state[p]['numel'],
"""

"""
        features = x.abs()
        to_log = [
            _iter,
            features.std().item(), # std
            features.mean().item(), # mean
            features.max().item(), # max
        ]
"""




"""
99
        vv = 4468
        dd = 30


investigating 95 16k b/16 2e-3.
notes:
1. spike
        vv = 7972
        dd = 1e1
vis.pos_embed > 2.2 rms std.
vis.pos_embed > 18 rms max
2. spike
        vv = 3280
        dd = 1e1
vis.pos_embed > 2.2 rms std.
vis.pos_embed > 17.5 rms max

"""