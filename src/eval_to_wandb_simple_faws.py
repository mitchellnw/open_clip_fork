import glob
import wandb

# send only new somehow
files = list(glob.glob("evals-faws/*.pt"))
print('here', files)
metrics = {}
for filename in files:
    if 'latest' in filename:
        continue
    # get name
    name = filename.split('/')[-1].split('---')[0]
    if name not in metrics:
        metrics[name] = []
    epoch = int(filename.split("_")[-1].split(".")[0])
    f = open(filename, "r")
    c = f.read()
    good =[l for l in c.split("\n") if "imagenet-zero" in l]
    if len(good) != 1:
        continue
    top5 = float(good[0].split("\t")[1].split(" ")[-1])
    top1 = float(good[0].split("\t")[0].split(" ")[-1])
    metrics[name].append([epoch, top1, top5])

for name in metrics:
    metrics[name].sort(key=lambda x:x[0])
    print(name, metrics)

for name in metrics:
    wandb.init(project="open_clip10", name=name, id=name, resume=True)

    for epoch, top1, top5 in metrics[name]:
        wandb.log({'top1': top1, 'top5': top5, 'step': epoch})

    wandb.finish()