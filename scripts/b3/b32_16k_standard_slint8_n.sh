#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=sopenclip
#SBATCH --nodes 2
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

module load openmpi
module load cuda/11.8

export MASTER_ADDR=`hostname`
export MASTER_PORT=12802
export NCCL_PROTO=simple
export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG=info

export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

cd /admin/home-mitchellw/forks/open_clip_fork/src
export PYTHONPATH="$PYTHONPATH:/admin/home-mitchellw/forks/open_clip_fork/src"

LR=2e-3
BETA2=0.98
MODEL=ViT-B-32
BS=16384
OPT=clipadamw

EXP_NAME="$OPT-slint8v5-$MODEL-$BS-$LR-$BETA2-v0"

srun --comment laion --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar" \
    --train-num-samples 65536000 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 5000 \
    --batch-size=1024 \
    --epochs=5 \
    --lr $LR \
    --beta2 $BETA2 \
    --workers=10 \
    --name ${EXP_NAME} \
    --logs /fsx/home-mitchellw/experimetns/opt3 \
    --model $MODEL \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --custom-attention vanilla \
    --save-most-recent \
    --advanced-logging \
    --wandb-project-name open_clip_12 \
    --force-patch-dropout 0.5 \
    --resume 'latest' \
    --slint8 \
    --precision custom_fp16 \
    --custom-scaler 65536 \
    --delete-previous-checkpoint \
    --opt $OPT

# info.
# 99 - 
# 98 - 
# 95 -
# 9 - 
# 8 - 
# 5 -