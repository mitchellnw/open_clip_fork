/private/home/mitchellw/miniconda3/envs/open_clip/bin/torchrun --nproc_per_node 2 -m training.main \
    --train-data '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 10 \
    --precision amp \
    --workers 4 --model ViT-B/32 \
    --imagenet-val /datasets01/imagenet_full_size/061417/val

wandb login --relogin --host https://api.wandb.ai
8c7af12c0467de2ba16c0109d7666b00b0baea89

srun --gpus-per-node=8 --nodes=1 --partition=devlab --time=72:00:00 -C volta32gb --cpus-per-task 48 --pty /bin/bash -l
/private/home/mitchellw/miniconda3/envs/open_clip/bin/torchrun --nproc_per_node 8 -m training.main  --ddp-static-graph --local-loss --dataset-resampled --gather-with-grad --grad-checkpointing   --train-data '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'     --train-num-samples 10968539     --dataset-type webdataset    --precision amp     --workers 4 --model ViT-H/14    --imagenet-val /datasets01/imagenet_full_size/061417/val --batch-size 16

/private/home/mitchellw/miniconda3/envs/open_clip/bin/torchrun --nproc_per_node 8 -m training.main  --ddp-static-graph --local-loss --dataset-resampled --gather-with-grad --grad-checkpointing   --train-data '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'     --train-num-samples 25600    --dataset-type webdataset    --precision amp     --workers 4 --model ViT-B/32  --batch-size 64 --report-to wandb --name wdsdebug2 

# vit l runs 512 for 32gb, 256 for 16gb
# vit h 128 / ?
# vit g 32 / ?