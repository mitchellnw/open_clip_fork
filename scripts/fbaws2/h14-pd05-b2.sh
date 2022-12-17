#!/bin/bash
#SBATCH --partition=learnlab
#SBATCH --job-name=openclip
#SBATCH --nodes 14
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --time=4320
#SBATCH --exclude=a100-st-p4d24xlarge-477

# can get up to 320
# 16 * 256 is right
# 14 * 292

export MASTER_PORT=12802

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

cd /fsx-labs/mitchellw/open_clip_fork/src
export PYTHONPATH="$PYTHONPATH:/fsx-labs/mitchellw/open_clip_fork/src"

EXP_NAME="clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-v1"

srun --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --report-to wandb \
    --train-data "/fsx-w3/akadian/laion2B-cvpr-filtered/shards/laion2B-en-joined{0..127}/{00000..00362}.tar" \
    --train-num-samples 100000000 \
    --warmup 8000 \
    --batch-size 292 \
    --dataset-type webdataset \
    --epochs 12 \
    --workers 4 \
    --model ViT-H-14-pd05 \
    --seed 0 \
    --lr 1e-3 \
    --beta2 0.95 \
    --name ${EXP_NAME} \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
    --logs "/fsx-labs/mitchellw/experiments/openclip2" \
    --advanced-logging \
    --wandb-project-name open_clip10