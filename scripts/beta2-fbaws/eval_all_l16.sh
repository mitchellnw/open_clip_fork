#!/bin/sh
# export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=12802
cd /fsx-labs/mitchellw/open_clip_fork/src

for i in `ls -t  /fsx-scaling/mitchellw/experiments/open_clip_b2/*/checkpoints/*6.pt`
do
    # model=${i:52:8}
    IFS='-' read -r -a array <<< "$i"
    model="${array[2]}-${array[3]}-${array[4]}"
    save_path="$(dirname $i)/eval.pt"
    if [ -f "$save_path" ]; then
        echo "$save_path exists."
    elif echo "$save_path" | grep -qE "sepcat|gadam|cadamw3cat"; then
        echo "$save_path does not exist.."
        torchrun --nproc_per_node 4 -m training.main \
        --batch-size 200   --workers 2 --model $model  --train-num-samples 413000000  \
        --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
        --save-most-recent --pretrained $i \
        --imagenet-val /datasets01/imagenet_full_size/061417/val --sep-attn &> $save_path
    else
        echo "$save_path does not exist."
        torchrun --nproc_per_node 4 -m training.main \
        --batch-size 200   --workers 2 --model $model  --train-num-samples 413000000  \
        --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
        --save-most-recent --pretrained $i \
        --imagenet-val /datasets01/imagenet_full_size/061417/val &> $save_path
    fi
done