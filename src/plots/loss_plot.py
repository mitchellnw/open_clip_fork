import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    kernel_size = 40
    min_loss = 10
    max_scaler = 0.25
    log_level = 2

    file_list = ['h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v0', 'h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1', 'h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v2', 'h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v1-try2']
    #file_list = [file_list[-1]]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
    for j, file in enumerate(file_list):
        for i in range(1):
            df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/loss.csv')
            ax1.plot(df.iloc[:, 0], np.minimum(min_loss, df.iloc[:, 1]), alpha=0.2, color=f'C{j}')
            
            kernel = np.ones(kernel_size) / kernel_size
            data_convolved = np.convolve(df.iloc[:, 1], kernel, mode='same')
            data_convolved = data_convolved[kernel_size:-kernel_size]
            ax1.plot(df.iloc[:, 0][kernel_size:-kernel_size], np.minimum(min_loss, data_convolved), color=f'C{j}')

            #ax1.set_ylim([0, 2])
        for i in range(1):
            df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/amp.csv')
            ax2.plot(df.iloc[:, 0], np.maximum(max_scaler, df.iloc[:, 1]))#, alpha=0.2)
            ax2.set_yscale('log')


        for i in range(1):
            df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/{file}/data/{i}/amp.csv')
            ax2.plot(df.iloc[:, 0], np.maximum(max_scaler, df.iloc[:, 1]))#, alpha=0.2)
            ax2.set_yscale('log')

    for ax in [ax1, ax2]:
        ax.grid()
        #break
        #ax.set_xlim([25100, 25300])
        #ax.set_xlim([0, 20])
    plt.savefig('plots/loss_plot.png', bbox_inches='tight')
