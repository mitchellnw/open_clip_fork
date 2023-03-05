#!/bin/bash
#SBATCH --partition=learnlab,scaling_data_pruning,learnfair
#SBATCH --job-name=openclip
#SBATCH --nodes 32
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=10
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --time=4320
#SBATCH --exclude=a100-st-p4d24xlarge-477,a100-st-p4d24xlarge-820,a100-st-p4d24xlarge-707,a100-st-p4d24xlarge-879,a100-st-p4d24xlarge-426,a100-st-p4d24xlarge-437
#SBATCH --requeue

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

LR=1e-3
BETA2=0.99
MODEL=ViT-L-16-pd05
BS=65536

EXP_NAME="cat-$MODEL-$BS-$LR-$BETA2-v0"

srun --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --report-to wandb \
    --train-data "/fsx-w3/akadian/laion2B-cvpr-filtered/shards/laion2B-en-joined{0..127}/{00000..00362}.tar" \
    --train-num-samples 393216000 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 8000 \
    --batch-size=256 \
    --epochs=6 \
    --lr $LR \
    --beta2 $BETA2 \
    --workers=6 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /fsx-scaling//mitchellw/experiments/open_clip_b2 \
    --model $MODEL \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
    --advanced-logging \
    --wandb-project-name cat_beta2

# cd /fsx-labs/mitchellw/open_clip_fork
# conda activate open_clip 

# info.
# 99 - 406222
# 95 -404860
# 8 - running