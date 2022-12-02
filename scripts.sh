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

/private/home/mitchellw/miniconda3/envs/open_clip/bin/torchrun --nproc_per_node 2 -m training.main  --ddp-static-graph --local-loss --dataset-resampled --gather-with-grad --grad-checkpointing   --train-data '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'     --train-num-samples 25600    --dataset-type webdataset    --precision amp  --workers 4 --model ViT-B/32  --batch-size 64 --report-to wandb --name wdsdebug3

# vit l runs 512 for 32gb, 256 for 16gb
# vit h 128 / ?
# vit g 32 / ?


/private/home/mitchellw/miniconda3/envs/open_clip/bin/torchrun --nproc_per_node 2 -m training.main \
    --batch-size 128 \
    --precision amp_bfloat16 \
    --workers 4 --model ViT-B/32 \
    --imagenet-val /datasets01/imagenet_full_size/061417/val \
    --pretrained /private/home/mitchellw/epoch_10.pt


torchrun --nproc_per_node 4 -m training.main \
    --batch-size 128 \
    --precision amp_bfloat16 \
    --workers 4 --model ViT-B/32 --force-custom-text \
    --imagenet-val /p/scratch/ccstdl/gordon2/imagenet_val \
    --pretrained /p//wortsman1/open_clip/logs/B-32_400M_epochs-32_precision-bfloat16/B-32_run-force-text-resume/checkpoints/epoch_32.pt


torchrun --nproc_per_node 4 -m training.main   \
      --batch-size 208   --workers 2 --model ViT-B/32     --dataset-type webdataset   \
      --train-data="pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/{00000..41455}.tar -"  \
      --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing \
      --precision amp_bfloat16


## checkpoint analyzer script
/private/home/mitchellw/miniconda3/envs/open_clip/bin/torchrun --nproc_per_node 2 -m training.main \
    --train-data '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 10 \
    --precision amp \
    --workers 4 --model ViT-H-14 \
    --seed 1000 \
    --resume /checkpoint/mitchellw/experiments/open_clip/clip-h14-400m-l0-opt-0.0005-0.9-0.98-1e-06-bs-8192-amp-v0/checkpoints/iter_73296.pt \
    --ddp-static-graph --local-loss --dataset-resampled --gather-with-grad --grad-checkpointing \
    --name cp_analyzer
    