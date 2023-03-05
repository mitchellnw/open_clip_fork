#!/bin/bash
#SBATCH --partition=g80n140
#SBATCH --job-name=sopenclip
#SBATCH --nodes 64
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --output=%x_%j.out
#SBATCH --comment=openclip
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --exclude ip-26-0-166-164,ip-26-0-162-75,ip-26-0-166-61,ip-26-0-162-33

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

EXP_NAME="bigG14-unmasked-tuning-v4"

#/opt/slurm/sbin/srun --comment openclip --cpu_bind=v --accel-bind=gn python -m training.main \
#torchrun --nproc_per_node 8 -m training.main \
srun --comment openclip --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar" \
    --train-num-samples 135646078 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 13000 \
    --batch-size=156 \
    --epochs=272 \
    --lr 4e-3 \
    --beta2 0.95 \
    --workers=4 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs /fsx/home-mitchellw/experimetns/open_clip/ \
    --model ViT-bigG-14-ls1 \
    --seed 400000 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --save-most-recent \
    --advanced-logging \
    --wandb-project-name open_clip6 \
    --accum-freq 2 \
    --resume "/fsx/home-mitchellw/experimetns/open_clip/clip-bigG14-pd05-ls1-pinit-160k-2e-3-0.95-amp_bfloat16-v1/checkpoints/epoch_256.pt"


# --train-data="pipe:aws s3 cp s3://deep-floyd-s3/datasets/{laion_cleaned-part1/{00000..79752}.tar,laion_cleaned-part2/{00000..94330}.tar,laion_cleaned-part3/{00000..94336}.tar,laion_cleaned-part4/{00000..94340}.tar,laion_cleaned-part5/{00000..94333}.tar,laion_cleaned-part6/{00000..77178}.tar} -" \


# aws s3 cp s3://s-laion/mitchellw/experiments/bigG14-unmasked-tuning/checkpoints/epoch_267.pt .
# aws s3 cp s3://s-laion/mitchellw/experiments/bigG14-unmasked-tuning/checkpoints/epoch_271.pt .
# aws s3 cp s3://s-laion/mitchellw/experiments/bigG14-unmasked-tuning/checkpoints/epoch_272.pt .