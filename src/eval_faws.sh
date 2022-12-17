# export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
# export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_PORT=12802

ev=eval_$2---`basename $1`
if [ -f "evals-faws/$ev" ]; then
  true
elif [[ "$ev" == *"latest"* ]]; then
  echo "Skipping latest."
else
  echo "$ev does not exist."
  echo "about to start eval."
  torchrun --nproc_per_node 4 -m training.main \
   --batch-size 200   --workers 2 --model ViT-H-14  --train-num-samples 413000000  \
   --local-loss  --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16  \
   --save-most-recent --pretrained $1 \
   --imagenet-val /datasets01/imagenet_full_size/061417/val &> evals-faws/$ev
  echo "done eval."
  python eval_to_wandb_simple_faws.py
fi