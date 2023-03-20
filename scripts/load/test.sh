

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
--local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
--save-most-recent --pretrained epoch_latest.pt \
--imagenet-val /fsx/rom1504/imagenetval/imagenet_validation