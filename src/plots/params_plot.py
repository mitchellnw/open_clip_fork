import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

    for i in [0, 8, 16]:
        df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/b32-400m-w-opt-0.001-0.9-0.98-1e-06-bs-4096-amp-v1/data/{i}/params-module.visual.transformer.resblocks.0.attn.in_proj_bias.csv')
        ax1.plot(df.iloc[:, 0], df.iloc[:, 1], label=i)#, alpha=0.2)
        ax2.plot(df.iloc[:, 0], df.iloc[:, 2], label=i)#, alpha=0.2)
    
    plt.savefig('plots/params_plot.png', bbox_inches='tight')
