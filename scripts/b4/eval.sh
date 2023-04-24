#!/bin/sh
# export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=12802
cd /admin/home-mitchellw/forks/open_clip_fork/src

for i in `ls -t  /fsx/home-mitchellw/experimetns/opt4/*/checkpoints/epoch*.pt`
do
    # model=${i:52:8}
    IFS='-' read -r -a array <<< "$i"
    model="${array[2]}-${array[3]}-${array[4]}"

    if [ "${model:0:1}" != "V" ]; then
        model="${array[3]}-${array[4]}-${array[5]}"
    fi

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
        # rm "$(dirname $i)/epoch_1.pt"
        # rm "$(dirname $i)/epoch_2.pt"
        # rm "$(dirname $i)/epoch_3.pt"
        # rm "$(dirname $i)/epoch_4.pt"
        # rm "$(dirname $i)/epoch_latest.pt"
        # if [[ $save_path == *"int8"* ]]; then
        #     rm $save_path
        #     echo $save_path
        # fi
        echo "$save_path exists."
    elif [[ $save_path == *"latest"* ]]; then
        echo "pass on latest"
    else
        if [[ $save_path == *"sglint8"* ]]; then
            echo "$save_path does not exist -- sglint8."
            torchrun --nproc_per_node 2 -m training.main \
            --batch-size 200   --workers 4 --model $model  --train-num-samples 413000000  \
            --local-loss  --gather-with-grad     --grad-checkpointing       --precision custom_fp16  \
            --save-most-recent --pretrained $i --custom-attention vanilla --sglint8 \
            --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $save_path
        elif [[ $save_path == *"slint8"* ]]; then
            echo "$save_path does not exist -- slint8."
            torchrun --nproc_per_node 2 -m training.main \
            --batch-size 200   --workers 4 --model $model  --train-num-samples 413000000  \
            --local-loss  --gather-with-grad     --grad-checkpointing       --precision custom_fp16  \
            --save-most-recent --pretrained $i --custom-attention vanilla --slint8 \
            --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $save_path
        elif [[ $save_path == *"int8"* ]]; then
            echo "$save_path does not exist -- int8."
            torchrun --nproc_per_node 2 -m training.main \
            --batch-size 200   --workers 4 --model $model  --train-num-samples 413000000  \
            --local-loss  --gather-with-grad     --grad-checkpointing       --precision custom_fp16  \
            --save-most-recent --pretrained $i --custom-attention vanilla --int8mix \
            --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $save_path
        elif [[ $save_path == *"fp8"* ]]; then
            echo "$save_path does not exist -- fp8."
            torchrun --nproc_per_node 2 -m training.main \
            --batch-size 200   --workers 4 --model $model  --train-num-samples 413000000  \
            --local-loss  --gather-with-grad     --grad-checkpointing       --precision custom_fp16  \
            --save-most-recent --pretrained $i --custom-attention vanilla --fp8mix \
            --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $save_path
        elif [[ $save_path == *"extraln"* ]]; then
            echo "$save_path does not exist -- extraln."
            torchrun --nproc_per_node 2 -m training.main \
            --batch-size 200   --workers 4 --model $model  --train-num-samples 413000000  \
            --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
            --save-most-recent --pretrained $i --custom-attention extra_ln \
            --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $save_path
        else
            echo "$save_path does not exist."
            torchrun --nproc_per_node 2 -m training.main \
            --batch-size 200   --workers 4 --model $model  --train-num-samples 413000000  \
            --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
            --save-most-recent --pretrained $i --custom-attention vanilla \
            --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $save_path
        fi
    fi
done

# """
# torchrun --nproc_per_node 2 -m training.main \
# --batch-size 200   --workers 4 --model ViT-L-14  --train-num-samples 413000000  \
# --local-loss  --gather-with-grad     --grad-checkpointing  --precision custom_fp16 --int8mix \
# --save-most-recent \
# --pretrained /fsx/home-mitchellw/experimetns/opt3/clipadamw-int8mix-ViT-L-14-16384-2e-3-0.98-v0/checkpoints/epoch_5.pt \
# --custom-attention vanilla \
# --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation


# 54.8
# 57.0
# """