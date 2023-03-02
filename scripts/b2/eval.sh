#!/bin/sh
# export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=12802
cd /admin/home-mitchellw/forks/open_clip_fork/src

for i in `ls -t  /fsx/home-mitchellw/experimetns/opt/*/checkpoints/*5.pt`
do
    # model=${i:52:8}
    IFS='-' read -r -a array <<< "$i"
    model="${array[2]}-${array[3]}-${array[4]}"

    if [ "${model:0:1}" != "V" ]; then
        model="${array[3]}-${array[4]}-${array[5]}"
    fi

    #echo $model
    save_path="$(dirname $i)/eval.pt"

    last=$(tail -c2 $save_path | head -c1)
    if [ "$last" = "=" ]; then
        echo "bad error with $save_path"
        rm $save_path
    fi

    if [ -f "$save_path" ]; then
        rm "$(dirname $i)/epoch_1.pt"
        rm "$(dirname $i)/epoch_2.pt"
        rm "$(dirname $i)/epoch_3.pt"
        rm "$(dirname $i)/epoch_4.pt"
        rm "$(dirname $i)/epoch_latest.pt"
        echo "$save_path exists."
    else
        if [[ $save_path == *"extraln"* ]]; then
            echo "$save_path does not exist -- extraln."
            torchrun --nproc_per_node 4 -m training.main \
            --batch-size 200   --workers 4 --model $model  --train-num-samples 413000000  \
            --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
            --save-most-recent --pretrained $i --custom-attention extra_ln \
            --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $save_path
        else
            echo "$save_path does not exist."
            torchrun --nproc_per_node 4 -m training.main \
            --batch-size 200   --workers 4 --model $model  --train-num-samples 413000000  \
            --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
            --save-most-recent --pretrained $i \
            --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $save_path
        fi
    fi
done
