/private/home/mitchellw/miniconda3/envs/open_clip/bin/torchrun --nproc_per_node 2 -m training.main \
    --train-data 'nvi/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar' \
    --train-num-samples 10968539 \
    --dataset-type webdataset \
    --batch-size 10 \
    --precision amp \
    --workers 4 \
    --imagenet-val /datasets01/imagenet_full_size/061417/val