#!/bin/bash -x

#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --exclude=jwb[0026,0098,0193,0631,0731,0729,0801,0807,0833,0964,1021]
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=12
# #SBATCH --wait-all-nodes=1
#SBATCH --time=2:00:00
#SBATCH --partition=develbooster
#SBATCH --job-name=clipavgeval


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


export CUDA_VISIBLE_DEVICES=0
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


cd /p/project/ccstdl/wortsman1/open_clip_fork/src

export CUDA_VISIBLE_DEVICES=0,1
for i in `ls -t  /p/project/ccstdl/wortsman1/open_clip_fork/logs/*/checkpoints/epoch*.pt`
do
    model='ViT-B-32'

    #echo $model
    save_path="$(dirname $i)/eval_$(basename $i)"

    if [ -f "$save_path" ]; then
        last=$(tail -c2 $save_path | head -c1)
        if [ "$last" = "=" ]; then
            echo "bad error with $save_path"
            rm $save_path
        fi
    fi

    if [ -f "$save_path" ]; then
        echo "$save_path exists."
    elif [[ $save_path == *"latest"* ]]; then
        echo "pass on latest"
    else
        torchrun --nproc_per_node 2 -m training.main \
        --batch-size 200   --workers 2 --model $model  --train-num-samples 413000000  \
        --local-loss  --gather-with-grad     --grad-checkpointing   --name $RANDOM    --precision amp_bfloat16  \
        --save-most-recent --pretrained $i \
        --imagenet-val /p/scratch/ccstdl/gordon2/imagenet_val &> $save_path
    fi
done

