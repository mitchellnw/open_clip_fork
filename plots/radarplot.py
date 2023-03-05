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


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarAxes(PolarAxes):

        name = 'radar'
        # use 1 line segment to connect specified points
        RESOLUTION = 1

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

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

    #fig = plt.figure(figsize=(12,8))
    #order = df.sort_values(by="dataset_type").dataset.unique()
    order = list(dataset_type.keys())
    order = order[:29]


    #order = order[:28]
    # ax = sns.barplot(
    #     x="dataset", y="acc1", 
    #     data=df,
    #     order=order,
    #     hue="model"
    # )
    # ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    # ax.set_axisbelow(True)
    # ax.yaxis.grid(color='gray')
    
    # df.to_csv('plots/clipbenchresults.csv')


    # plt.savefig('plots/radarplot.png', bbox_inches='tight')

    # scatter plots
    fig = plt.figure(constrained_layout=True, figsize=(12, 8))
    gs = GridSpec(1, 1, figure=fig)#, width_ratios=[2,2,4])


    colors = ['C0', 'C1']


    theta = radar_factory(len(order), frame='polygon')


    ax = fig.add_subplot(gs[0, 0], projection='radar')

    # ax.set_xlabel(f'Model size (GMACS, log scale)', fontsize=13)
    # ax.set_ylabel(f'Average accuracy', fontsize=13)
    # ax.set_title(f'Scaling multi-task patching', fontsize=15)

    xs = df.dataset.values
    #ys = 
    ys_h = []
    ys_g = []
    for o in order:
        ys_h.append(df[(df.dataset == o) & (df.model == 'ViT-H/14')].acc1.values[0])
        ys_g.append(df[(df.dataset == o) & (df.model == 'ViT-G/14')].acc1.values[0])


    #ax.plot(xs, ys_h, label=order)
    order[-1] = 'vtab/clevr_closest'
    xs = order

    ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
    ax.minorticks_off()
    ax.set_xticks([10, 20, 40, 80])
    ax.grid(which='major', axis='both', zorder=-1.0, alpha=0.3)


    train = ','.join(order)
    for i, d in [(0, ys_g), (1, ys_h)]:

        model = 'ViT-G/14' if i == 0 else 'ViT-H/14'
        ax.plot(theta, d, color=colors[i], label=model, linewidth=2,)#, marker=markers[mode], markersize=5)
        if i == 1:
            ax.fill(theta, d, facecolor=colors[i], alpha=0.2)
        else:
            d_small = ys_h
            print(len(d), len(d_small), len(theta))
            ax.fill_between(theta, d_small, d, facecolor=colors[i], alpha=0.2)
            ax.fill_between([theta[0], theta[-1]], [d_small[0], d_small[-1]], [d[0], d[-1]], facecolor=colors[i], alpha=0.2)
        ax.set_varlabels(order)
        ax.tick_params(axis='both', zorder=50)
        
        xticks = ax.xaxis.get_major_ticks()
        # xticks[3].set_pad(15)
        # xticks[7].set_pad(15)
        xticks[0].set_pad(10)

    ax.legend(bbox_to_anchor=[1.1, 1.05],)

    plt.savefig('plots/radarplot.png', bbox_inches='tight')