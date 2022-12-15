torchrun --nproc_per_node 2 -m training.main   \
      --batch-size 10   --workers 2 --model ViT-B/32     --dataset-type webdataset   \
      --train-data="/fsx-w3/akadian/laion2B-cvpr-filtered/shards/laion2B-en-joined{0..127}/{00000..00362}.tar"  \
      --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing \
      --precision amp_bfloat16 --cinit --name test123 --save-most-recent


cd /fsx-labs/mitchellw/open_clip_fork/src/
export PYTHONPATH=$PWD

srun --gres=gpu:4 --cpus-per-task 6 --partition=scaling_data_pruning --time=1:00:00 --pty /bin/bash -l