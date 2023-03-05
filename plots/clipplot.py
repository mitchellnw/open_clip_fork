import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import matplotlib.pyplot as plt
import matplotlib.font_manager
import matplotlib as mpl
import seaborn as sns




if __name__ == '__main__':
    pd.options.display.float_format = '{:.3f}'.format
    def extract_arch(model):
        vit, size, patch_size, *rest = model.split("-")
        return vit+"-"+size+"-"+patch_size
    plt.rcParams['figure.dpi'] = 200

    dataset_type = pd.read_csv("plots/dataset_type.csv").set_index("dataset")["type"].to_dict()
    df = pd.read_csv("plots/romresults.csv")
    vtab_plus = list(map(lambda s:s.strip(), open("plots/datasets.txt").readlines()))
    df = df[df.dataset.isin(vtab_plus)]
    df.loc[:, "dataset_type"] = df.dataset.apply(lambda d:dataset_type[d])
    df.loc[:, "model_arch"] = df.model.apply(extract_arch)

    df_retrieval = df[df["dataset_type"] == "retrieval"]
    df = df[df["dataset_type"] != "retrieval"]
    df = df.drop(["image_retrieval_recall@5", "text_retrieval_recall@5"], axis=1)
    dataset_type = {k:v for k,v in dataset_type.items() if v != "retrieval"}

    df["model_fullname"] = df["model_fullname"].str.replace("/fsx/rom1504/open_clip/good_models/", "")
    df["model_fullname"] = df["model_fullname"].str.replace("/fsx/rom1504/CLIP_benchmark/benchmark/", "")
    df["model_fullname"] = df["model_fullname"].str.replace(".pt", "")
    df["model_fullname"] = df["model_fullname"].str.replace("ViT-H-14 h_256", "ViT-H/14")
    df["model_fullname"] = df["model_fullname"].str.replace("ViT-bigG-14 bigG_14_v2_sd", "ViT-G/14")
    df["model"] = df["model_fullname"]
    df["pretrained"] = df["pretrained"].str.replace("/fsx/rom1504/open_clip/good_models/", "")
    df["pretrained"] = df["pretrained"].str.replace("/fsx/rom1504/CLIP_benchmark/benchmark/", "")


    order = list(dataset_type.keys())


    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = GridSpec(1, 1, figure=fig)#, width_ratios=[2,2,4])


    colors = ['C0', 'C1']

    ax = fig.add_subplot(gs[0, 0])


    xs = df.dataset.values

    ys_h = []
    ys_g = []
    diff = []
    for o in order:
        ys_h.append(df[(df.dataset == o) & (df.model == 'ViT-H/14')].acc1.values[0])
        ys_g.append(df[(df.dataset == o) & (df.model == 'ViT-G/14')].acc1.values[0])
        diff.append(df[(df.dataset == o) & (df.model == 'ViT-G/14')].acc1.values[0] - df[(df.dataset == o) & (df.model == 'ViT-H/14')].acc1.values[0])

    
    
    diff, order = zip(*sorted(zip(diff, order)))
    diff = list(diff)
    order = list(order)
    # diff.reverse()
    # order.reverse()
    cols = []
    for d in diff:
        if d >= 0:
            cols.append('C2')
        else:
            cols.append('C3')
    ax.barh(order, diff, align='center', color=cols)

    ax.text(0.25, 15, 'Comparison to previous open source SoTA\n\nΔ Zero-shot accuracy (percentage points)\nOpenCLIP ViT-G/14 vs. ViT-H/14', ha = 'center', va = 'center', fontsize=20)
    ax.set_axisbelow(True)
    ax.xaxis.grid(color='gray')

    ax.spines[['right', 'top', 'bottom']].set_visible(False)
    ax.set_xticks([])

    for idx, (bar, val) in enumerate(zip(ax.patches, diff)):
        if val > 0:
            pos = val + 0.001
            color = 'gray'
            ax.text(pos, bar.get_y()+bar.get_height()/2, f'+{100*val:.1f}', color = color, ha = 'left', va = 'center', fontsize=10)
        else:
            pos = 0.001
            color = 'gray'
            ax.text(pos, bar.get_y()+bar.get_height()/2, f'{100*val:.1f}', color = color, ha = 'left', va = 'center', fontsize=10)   

    plt.savefig('plots/clipplot.png', bbox_inches='tight')