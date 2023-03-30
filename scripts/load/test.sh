

torchrun --nproc_per_node 8 -m training.main   \
      --batch-size 208   --workers 2 --model ViT-B-32     --dataset-type webdataset   \
      --train-data="pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/{00000..41455}.tar -"  \
      --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing \
      --precision amp_bfloat16 --custom-attention vanilla --opt customadamw --force-patch-dropout 0.5 \
      --rms-load /fsx/home-mitchellw/experimetns/opt3/customadamw-ViT-B-32-8192-2e-3-0.995-rmscheck-v0/spike/step_4659 \
      --resume /fsx/home-mitchellw/experimetns/opt3/customadamw-ViT-B-32-8192-2e-3-0.995-rmscheck-v0/spike/step_4659/model.pt





torchrun --nproc_per_node 8 -m training.main   \
      --batch-size 208   --workers 2 --model ViT-B-32     --dataset-type webdataset   \
      --train-data="pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/{00000..41455}.tar -"  \
      --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing \
      --precision amp_bfloat16 --custom-attention vanilla --opt customadamw --force-patch-dropout 0.5 \
      --rms-load /fsx/home-mitchellw/experimetns/opt3/customadamw-ViT-B-32-8192-2e-3-0.995-rmscheck-v1/spike/step_4443 \
      --resume /fsx/home-mitchellw/experimetns/opt3/customadamw-ViT-B-32-8192-2e-3-0.995-rmscheck-v1/spike/step_4443/model.pt


torchrun --nproc_per_node 2 -m training.main \
      --batch-size 200   --workers 4 --model ViT-L-14  --train-num-samples 413000000  \
      --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp  \
      --save-most-recent \
      --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation

# 55.11

rm -rf logs && torchrun --nproc_per_node 4 -m training.main   \
      --batch-size 256   --workers 8 --model ViT-L-14     --dataset-type webdataset   \
      --train-data="pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/{00000..41455}.tar -"  \
      --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing \
      --precision amp --custom-attention vanilla  \
      --sglint8 --log-every-n-steps 1





### BATCH 512

### BATCH 256


# H
# 392.4 samples/s sglint8
# 294.3 samples/s autogradlinear
# 339.8 samples/s nnlinear

# L
# 735.1 samples/s sglint8
# 599.9 autogradlinear
# 683.1 samples/s nnlinear

# B-16
# 2090 
# 1830
# 2085 

### BATCH 128


# aws s3 ls s3://deep-floyd-s3/datasets/laion_cleaned-part6/