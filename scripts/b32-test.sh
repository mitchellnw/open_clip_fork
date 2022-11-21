#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=openclip
#SBATCH --nodes 20
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=6
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --comment=openclip
#SBATCH --open-mode=append
#SBATCH --exclusive

module load intelmpi
source /opt/intel/mpi/latest/env/vars.sh
# export LD_LIBRARY_PATH=/opt/aws-ofi-nccl/lib:/opt/amazon/efa/lib64:/usr/local/cuda-11.0/efa/lib:/usr/local/cuda-11.0/lib:/usr/local/cuda-11.0/lib64:/usr/local/cuda-11.0:/opt/nccl/build/lib:/opt/aws-ofi-nccl-install/lib:/opt/aws-ofi-nccl/lib:$LD_LIBRARY_PATH
# export NCCL_PROTO=simple
# export PATH=/opt/amazon/efa/bin:$PATH
# export LD_PRELOAD="/opt/nccl/build/lib/libnccl.so"

# export FI_EFA_FORK_SAFE=1
# export FI_LOG_LEVEL=1
# export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

#export NCCL_ALGO=ring
export NCCL_DEBUG=info
#export NCCL_DEBUG_SUBSYS=INIT,ENV,GRAPH,COLL

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
# export OMPI_MCA_mtl_base_verbose=1
# export FI_EFA_ENABLE_SHM_TRANSFER=0
# export FI_PROVIDER=efa
# export FI_EFA_TX_MIN_CREDITS=64
# export NCCL_TREE_THRESHOLD=0


#export NCCL_P2P_DISABLE=1
#export NCCL_IBEXT_DISABLE=1
#export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"

# sent to sub script
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

echo go $COUNT_NODE
echo $HOSTNAMES

#source /fsx/rom1504/open_clip/.env/bin/activate
cd /fsx/home-mitchellw/open_clip_fork/src
export PYTHONPATH="$PYTHONPATH:/fsx/home-mitchellw/open_clip_fork/src"


EXP_NAME="b32-test-3"
LOGS_NAME="b32-test-3"

/opt/slurm/sbin/srun --comment laion --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --train-data="pipe:aws s3 cp s3://s-datasets/laion5b/laion2B-data/{000000..231349}.tar -" \
    --train-num-samples 135646078 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 2000 \
    --batch-size=200 \
    --epochs=256 \
    --lr 5.5e-4 \
    --workers=2 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs logs/${LOGS_NAME} \
    --model ViT-H-14 \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp \
    --save-most-recent \


{"dataset": "imagenet1k-unverified", "model": "ViT-H-14", "pretrained": "laion2b_s32b_b79k", "task": "zeroshot_classification", "metrics": {"acc1": 0.78942, "acc5": 0.95784, "mean_per_class_recall": 0.7895599999999999}, "language": "en"}

python clip_benchmark/cli.py --dataset=imagenet1k-unverified --dataset_root ~/imagenet --task=zeroshot_classification --pretrained=frozen_laion5b_s13b_b90k --model=xlm-roberta-large-ViT-H-14 --output=result.json --cupl --batch_size=128 --save_clf roberta_cupl.pt
python clip_benchmark/cli.py --dataset=imagenet1k-unverified --dataset_root ~/imagenet --task=zeroshot_classification --pretrained=frozen_laion5b_s13b_b90k --model=xlm-roberta-large-ViT-H-14 --output=result.json --batch_size=128 --save_clf roberta_standard.pt
python clip_benchmark/cli.py --dataset=imagenet1k-unverified --dataset_root ~/imagenet --task=zeroshot_classification --pretrained=laion2b_s32b_b79k --model=ViT-H-14 --output=result.json --cupl --batch_size=128 --save_clf standard_cupl.pt
python clip_benchmark/cli.py --dataset=imagenet1k-unverified --dataset_root ~/imagenet --task=zeroshot_classification --pretrained=laion2b_s32b_b79k --model=ViT-H-14 --output=result.json --batch_size=128 --save_clf standard_standard.pt

python clip_benchmark/cli.py --dataset=imagenet1k-unverified --dataset_root ~/imagenet --task=zeroshot_classification --pretrained=laion2b_s32b_b79k --model=ViT-H-14 --output=result.json --batch_size=128 --load_clfs  standard_cupl.pt standard_standard.pt


