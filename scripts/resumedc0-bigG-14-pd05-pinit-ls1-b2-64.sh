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
#SBATCH --exclude ip-26-0-170-129,ip-26-0-160-5,ip-26-0-163-9,ip-26-0-166-121,ip-26-0-166-164,ip-26-0-162-75,ip-26-0-166-61,ip-26-0-162-33
# #SBATCH -w ip-26-0-160-5,ip-26-0-162-181,ip-26-0-162-210,ip-26-0-162-226,ip-26-0-162-238,ip-26-0-163-9,ip-26-0-163-129,ip-26-0-163-220,ip-26-0-163-225,ip-26-0-163-226,ip-26-0-163-245,ip-26-0-163-251,ip-26-0-164-29,ip-26-0-164-80,ip-26-0-164-121,ip-26-0-165-144,ip-26-0-165-179,ip-26-0-166-3,ip-26-0-166-55,ip-26-0-166-107,ip-26-0-166-108,ip-26-0-166-121,ip-26-0-166-143,ip-26-0-166-169,ip-26-0-166-252,ip-26-0-167-61,ip-26-0-167-83,ip-26-0-167-150,ip-26-0-167-178,ip-26-0-168-172,ip-26-0-169-64,ip-26-0-169-95,ip-26-0-169-118,ip-26-0-169-140,ip-26-0-169-143,ip-26-0-169-172,ip-26-0-171-4,ip-26-0-171-5,ip-26-0-171-8,ip-26-0-171-43,ip-26-0-171-57,ip-26-0-171-135,ip-26-0-171-150,ip-26-0-171-191,ip-26-0-171-195,ip-26-0-171-247,ip-26-0-172-226,ip-26-0-173-9,ip-26-0-173-35,ip-26-0-173-49,ip-26-0-173-59,ip-26-0-173-61,ip-26-0-173-76,ip-26-0-173-96,ip-26-0-173-98,ip-26-0-173-114,ip-26-0-175-68,ip-26-0-175-80,ip-26-0-175-90,ip-26-0-175-107,ip-26-0-175-113,ip-26-0-175-145,ip-26-0-175-194,ip-26-0-175-210

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

cd /admin/home-mitchellw/git/open_clip_fork/src
export PYTHONPATH="$PYTHONPATH:/admin/home-mitchellw/git/open_clip_fork/src"

EXP_NAME="bigG14-unmasked-tuning-dc1"

#/opt/slurm/sbin/srun --comment openclip --cpu_bind=v --accel-bind=gn python -m training.main \
#torchrun --nproc_per_node 8 -m training.main \
/fsx/good_srun.sh --comment openclip --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="pipe:aws s3 cp s3://s-laion/datanet_7B_pool-baselines/clip_threshold/clip_l14_f0.3/seed_0/reshards_n67/4/{00000000..00004456}.tar::s3://s-laion/datanet_7B_pool-baselines/clip_threshold/clip_l14_f0.3/seed_0/reshards_n67/5/{00000000..00004462}.tar -" \
    --train-num-samples 135646078 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 13000 \
    --batch-size=156 \
    --epochs=288 \
    --lr 5e-5 \
    --beta2 0.95 \
    --workers=12 \
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
    --wandb-project-name open_clip6 \
    --accum-freq 2 \
    --resume "/fsx/home-mitchellw/bigG_14_e256_80.pt"


# --train-data="pipe:aws s3 cp s3://deep-floyd-s3/datasets/{laion_cleaned-part1/{00000..79752}.tar,laion_cleaned-part2/{00000..94330}.tar,laion_cleaned-part3/{00000..94336}.tar,laion_cleaned-part4/{00000..94340}.tar,laion_cleaned-part5/{00000..94333}.tar,laion_cleaned-part6/{00000..77178}.tar} -" \