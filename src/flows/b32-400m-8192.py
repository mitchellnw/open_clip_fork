import argparse
import os


from run_with_submitit import main_with_args, parse_args


if __name__ == "__main__":

    

    args = parse_args()

    args.ngpus = 8
    args.batch_size = 256
    args.nodes = 4

    name = f'b32-400m-bs-{args.batch_size * args.ngpus * args.nodes}'

    args.partition = 'devlab'
    args.use_volta32 = True

    args.job_dir = name
    args.name = name

    args.imagenet_val = '/datasets01/imagenet_full_size/061417/val'
    args.train_data = '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'
    args.train_num_samples = 40000000
    args.dataset_type = 'webdataset'
    args.model = 'ViT-B/32'
    #args.batch_size = 64
    args.precision = 'amp'
    args.workers = 6
    args.lr = 1e-3
    args.epochs = 160
    args.report_to = 'wandb'
    args.seed = 0
    args.ddp_static_graph = True
    args.local_loss = True
    args.dataset_resampled = True
    args.gather_with_grad = True
    args.grad_checkpointing = True
    args.save_frequency = 1
    args.zeroshot_frequency = 1
    args.warmup = 10000

    main_with_args(args)

"""
srun --cpu_bind=none,v --accel-bind=gn python -u src/training/main.py \
    --save-frequency 1 \
    --zeroshot-frequency 1 \
    --train-data="/p/fastdata/mmlaion/laion2B-en/{00000..23295}.tar" \
    --train-num-samples=200000000 \
    --warmup 10000 \
    --lr "1e-3" \
    --batch-size=208 \
    --epochs=160 \
    --workers=6 \
    --model ViT-L-14 \
    --name "L14-laion2B" \
    --report-to "tensorboard" \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --dataset-resampled \
    --gather-with-grad \
    --grad-checkpointing
"""