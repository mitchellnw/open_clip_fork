import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

    fig = plt.figure(figsize=(12,8))
    #order = df.sort_values(by="dataset_type").dataset.unique()
    order = list(dataset_type.keys())
    
    ax = sns.barplot(
        x="dataset", y="acc1", 
        data=df,
        order=order,
        hue="model"
    )
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='gray')
    
    df.to_csv('plots/clipbenchresults.csv')


    plt.savefig('plots/barplot.pdf', bbox_inches='tight')