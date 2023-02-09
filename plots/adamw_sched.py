import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os


if __name__ == '__main__':
    kernel_size = 40
    min_loss = 14
    max_scaler = 1
    log_level = 1

    fig, axlist = plt.subplots(1, 1, figsize=(8, 5))
    axlist = [axlist]
    ax = axlist[0]

    xs = np.arange(1, 1000, 1)
    ys = []
    beta2 = 0.9
    for x in xs:
        t = x + 1
        ys.append( beta2 * (1 - beta2**(t - 1)) / (1 - beta2**t) )

    ax.plot(xs, ys)

    plt.savefig('/admin/home-mitchellw/forks/open_clip_fork/plots/adamw_sched.png', bbox_inches='tight')



