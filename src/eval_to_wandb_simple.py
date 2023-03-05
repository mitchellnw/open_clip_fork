import glob
import wandb
import json

# send only new somehow
files = list(glob.glob("eval5/eval_epoch_*.pt"))
metrics = []
for filename in files:
    if 'latest' in filename:
        continue
    epoch = int(filename.split("_")[-1].split(".")[0])
    f = open(filename, "r")
    c = f.read()
    good =[l for l in c.split("\n") if "imagenet-zero" in l]
    if len(good) != 1:
        continue
    top5 = float(good[0].split("\t")[1].split(" ")[-1])
    top1 = float(good[0].split("\t")[0].split(" ")[-1])
    metrics.append([epoch, top1, top5])

metrics.sort(key=lambda x:x[0])
print(metrics)

wandb.init(project="open_clip6", name="eval_bigG_unmask_try_3", id="eval_bigG_unmask_try_3", resume=True)

with open('latest_evals.json', 'w') as f:
    json.dump(metrics, f)

for epoch, top1, top5 in metrics:
    wandb.log({'top1': top1, 'top5': top5, 'step': epoch})
