#!/bin/bash
#SBATCH --partition=g80n140
#SBATCH --job-name=sopenclip
#SBATCH --nodes 78
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --exclude=ip-26-0-160-13

module load openmpi
# source /opt/intel/mpi/latest/env/vars.sh

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

cd /fsx/home-mitchellw/forks/open_clip_fork/src
export PYTHONPATH="$PYTHONPATH:/fsx/home-mitchellw/forks/open_clip_fork/src"

EXP_NAME="clip-bigG14-pd05-ls1-pinit-160k-2e-3-0.95-amp_bfloat16-v1"

srun --comment laion --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar -" \
    --train-num-samples 135646078 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 13000 \
    --batch-size=256 \
    --epochs=256 \
    --lr 2e-3 \
    --beta2 0.95 \
    --workers=2 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /fsx/home-mitchellw/experimetns/open_clip/ \
    --model ViT-bigG-14-pd05-ls1 \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
    --advanced-logging \
    --wandb-project-name open_clip6 \
    --pinit
