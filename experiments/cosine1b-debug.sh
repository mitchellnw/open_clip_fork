#!/bin/bash -x

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --exclude=jwb[0026,0098,0193,0631,0731,0729,0801,0807,0833,0964,1021]
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
# #SBATCH --wait-all-nodes=1
#SBATCH --time=02:00:00
#SBATCH --partition=develbooster
#SBATCH --job-name=clipavg


# load low-level libraries
ml purge


# CONDA_ENV="open_clip"
CONDA_ENV="py9"

source /p/project/ccstdl/wortsman1/miniconda3/bin/activate ${CONDA_ENV}


# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export TORCH_DISTRIBUTED_DEBUG=INFO

export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10


export CUDA_VISIBLE_DEVICES=0,1,2,3
export NCCL_ASYNC_ERROR_HANDLING=1

export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}

export MASTER_PORT=12802
### get the first node name as master address - customized for vgg slurm
### e.g. master(gnodee[2-5],gnoded1) == gnodee2
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr"i"
echo "MASTER_ADDR="$MASTER_ADDR


# OPEN_CLIP_HOME="/p/project/laionize/jitsev1_juwelsbooster/open_clip_lr_mod"
OPEN_CLIP_HOME="/p/project/ccstdl/wortsman1/open_clip_fork"
export PYTHONPATH="$PYTHONPATH:${OPEN_CLIP_HOME}/src"
# export PYTHONPATH="$PYTHONPATH:${HOME}/home/open_clip/src"

EXP_NAME="cosine1b"

cd ${OPEN_CLIP_HOME}

srun --cpu_bind=v --accel-bind=gn --threads-per-core=1 python -u -m training.main \
    --save-frequency 1 \
    --train-data="/p/fastdata/mmlaion/laion2B-en/{00000..23295}.tar" \
    --train-num-samples=135646078 \
    --dataset-resampled \
    --warmup 2000 \
    --batch-size=60 \
    --epochs=256 \
    --workers=2 \
    --report-to=tensorboard \
    --model ViT-B-32 \
    --force-patch-dropout 0.5 \
    --name ${EXP_NAME} \
    --logs logs/${EXP_NAME} \
    --seed 0 \
    --lr 1e-3 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --resume "latest" \
    --grad-clip-norm 1 \
    --averagers poly_8_1,poly_16_1