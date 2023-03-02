#!/bin/bash
#SBATCH --partition=g40423
#SBATCH --job-name=sopenclip
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=12
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --time=4320
# #SBATCH --exclude=a100-st-p4d24xlarge-825,a100-st-p4d24xlarge-477,a100-st-p4d24xlarge-820,a100-st-p4d24xlarge-707,a100-st-p4d24xlarge-879,a100-st-p4d24xlarge-426,a100-st-p4d24xlarge-437,a100-st-p4d24xlarge-451,a100-st-p4d24xlarge-461
#SBATCH --requeue
#SBATCH --comment laion

# can get up to 320
# 16 * 256 is right
# 14 * 292

export MASTER_PORT=12802

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

cd /admin/home-mitchellw/forks/open_clip_fork/src
export PYTHONPATH="$PYTHONPATH:/admin/home-mitchellw/forks/open_clip_fork/src"

LR=5e-4
BETA2=0.98
MODEL=ViT-H-14
BS=16384
OPT=lion

EXP_NAME="$OPT-$MODEL-$BS-$LR-$BETA2-wd1beta195-v0"

/opt/slurm/bin/srun --comment laion --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --report-to wandb \
    --train-data="s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar" \
    --train-num-samples 65536000 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 5000 \
    --batch-size=256 \
    --epochs=5 \
    --lr $LR \
    --beta2 $BETA2 \
    --workers=10 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /fsx/home-mitchellw/experimetns/opt3 \
    --model $MODEL \
    --seed 1 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
    --advanced-logging \
    --wandb-project-name open_clip_12 \
    --force-patch-dropout 0.5 \
    --resume 'latest' \
    --custom-attention vanilla \
    --wd 1.0 \
    --beta1 0.95 \
    --delete-previous-checkpoint \
    --opt $OPT

# info.
# 99 - 
# 98 - 
# 95 -
# 9 - 
# 8 - 
# 5 -