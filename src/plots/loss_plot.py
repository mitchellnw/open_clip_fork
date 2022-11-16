import matplotlib.pyplot as plt
import pandas as pd

if __name__ == '__main__':

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    for i in range(32):
        df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/b32-400m-w-opt-0.001-0.9-0.98-1e-06-bs-4096-amp-v1/data/{i}/loss.csv')
        ax1.plot(df.iloc[:, 0], df.iloc[:, 1])#, alpha=0.2)
    for i in range(32):
        df = pd.read_csv(f'/checkpoint/mitchellw/experiments/open_clip/b32-400m-w-opt-0.001-0.9-0.98-1e-06-bs-4096-amp-v1/data/{i}/amp.csv')
        ax2.plot(df.iloc[:, 0], df.iloc[:, 1])#, alpha=0.2)
    plt.savefig('plots/loss_plot.png', bbox_inches='tight')
