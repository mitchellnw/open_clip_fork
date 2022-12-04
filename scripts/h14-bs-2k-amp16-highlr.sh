#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=openclip
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --comment=openclip
#SBATCH --open-mode=append
#SBATCH --exclusive

module load intelmpi
source /opt/intel/mpi/latest/env/vars.sh

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

cd /fsx/home-mitchellw/open_clip_fork/src
export PYTHONPATH="$PYTHONPATH:/fsx/home-mitchellw/open_clip_fork/src"

EXP_NAME="clip-h14-400m-l0-opt-0.001-0.9-0.98-1e-06-bs-2048-amp_bfloat16-v0"
LOGS_NAME="clip-h14-400m-l0-opt-0.001-0.9-0.98-1e-06-bs-2048-amp_bfloat16-v0"

/opt/slurm/sbin/srun --comment openclip --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar -" \
    --train-num-samples 135646078 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 10000 \
    --batch-size=128 \
    --epochs=256 \
    --lr 1e-3 \
    --workers=2 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /fsx/home-mitchellw/experimetns/open_clip/${EXP_NAME} \
    --model ViT-H-14 \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
