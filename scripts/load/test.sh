

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
      --batch-size 256   --workers 8 --model ViT-H-14     --dataset-type webdataset   \
      --train-data="pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/{00000..41455}.tar -"  \
      --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing \
      --precision amp --custom-attention vanilla  \
      --log-every-n-steps 1 --sglint8 \





### BATCH 128

# H
# sb - 369.5
# llm.int8 - 170

# L 
# sb - 687
# llm.int8 - 306


# B 
# sb - 1887
# llm.int8 - 862


### BATCH 256


# H
# 392.4 samples/s sglint8 # 390.5
# 294.3 samples/s autogradlinear
# 339.8 samples/s nnlinear

# L
# 735.1 samples/s sglint8
# 599.9 autogradlinear
# 683.1 samples/s nnlinear
# llm.int8 - 330

# B-16
# 2090 
# 1830
# 2085 
# llm.int8 - 951

### BATCH 128


# SB
# 21 mem originally 