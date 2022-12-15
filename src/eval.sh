# export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=12802

ev=eval_`basename $1`
if [ -f "$ev" ]; then
  true
elif [[ "$ev" == *"latest"* ]]; then
  echo "Skipping latest."
else
  echo "$ev does not exist."
  echo "about to start eval."
  torchrun --nproc_per_node 4 -m training.main \
   --batch-size 200   --workers 2 --model ViT-bigG-14-ls1  --train-num-samples 413000000  \
   --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
   --save-most-recent --pretrained $1 \
   --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation &> $ev
  echo "done eval."
  python eval_to_wandb_simple.py
fi