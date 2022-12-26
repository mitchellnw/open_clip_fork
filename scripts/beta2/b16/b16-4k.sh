#!/bin/bash
#SBATCH --partition=g40423
#SBATCH --job-name=sopenclip
#SBATCH --nodes 2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
#SBATCH --comment=laion
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --exclude=ip-26-0-138-235,ip-26-0-138-237,ip-26-0-138-235,ip-26-0-138-237,ip-26-0-138-235,ip-26-0-138-237,ip-26-0-138-235,ip-26-0-138-237,ip-26-0-138-235,ip-26-0-138-237,ip-26-0-138-235,ip-26-0-138-237,ip-26-0-138-235,ip-26-0-138-237,ip-26-0-138-235,ip-26-0-138-237,ip-26-0-140-166,ip-26-0-140-169,ip-26-0-140-172,ip-26-0-140-183,ip-26-0-140-188,ip-26-0-140-189,ip-26-0-140-194,ip-26-0-140-206,ip-26-0-140-207,ip-26-0-140-208,ip-26-0-140-226,ip-26-0-140-244,ip-26-0-141-2,ip-26-0-141-11,ip-26-0-141-14,ip-26-0-141-23


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

LR=1e-3
BETA2=0.999
MODEL=ViT-B-16-pd05
BS=4096

EXP_NAME="$MODEL-$BS-$LR-$BETA2-v0"

srun --comment laion --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="pipe:aws s3 cp s3://deep-floyd-s3/datasets/{laion_cleaned-part1/{00000..79752}.tar,laion_cleaned-part2/{00000..94330}.tar,laion_cleaned-part3/{00000..94336}.tar,laion_cleaned-part4/{00000..94340}.tar,laion_cleaned-part5/{00000..94333}.tar,laion_cleaned-part6/{00000..77178}.tar} -" \
    --train-num-samples 49152000 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 8000 \
    --batch-size=256 \
    --epochs=3 \
    --lr $LR \
    --beta2 $BETA2 \
    --workers=6 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /fsx/home-mitchellw/experimetns/open_clip_b2/ \
    --model $MODEL \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
    --advanced-logging \
    --wandb-project-name beta2

# gpu_tester --nodes 2 --parallel-tests 50 --job_comment laion --partition "g40423" --test_kind "ddp" --job_timeout 45


# info.
# 999 - 10177
# 99 - 10176
# 95 - 10175
# 8 - 10174
