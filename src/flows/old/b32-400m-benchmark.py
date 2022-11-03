import argparse
import os
from training.params import get_default_params

from run_with_submitit import main_with_args, parse_args


if __name__ == "__main__":

    args = parse_args()

    args.model = 'ViT-B/32'
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
            print('setting default', name, val)

    args.ngpus = 8
    args.batch_size = 128
    args.nodes = 1
    args.lr = 1e-3

    args.partition = 'learnlab'
    args.use_volta32 = False

    args.imagenet_val = '/datasets01/imagenet_full_size/061417/val'
    args.train_data = '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'
    args.train_num_samples = 40000000
    args.dataset_type = 'webdataset'
    
    args.precision = 'amp'
    args.workers = 6
    
    args.epochs = 160
    args.report_to = 'wandb'
    args.seed = 1
    args.ddp_static_graph = True
    args.local_loss = True
    args.dataset_resampled = True
    args.gather_with_grad = True
    args.grad_checkpointing = True
    args.save_frequency = 1
    args.zeroshot_frequency = 1
    args.warmup = 10000

    name = f'benchmark2-{args.nodes}-b32-400m-opt-{args.lr}-{args.beta1}-{args.beta2}-{args.eps}-bs-{args.batch_size * args.ngpus * args.nodes}-{args.precision}-s{args.seed}'
    if os.path.exists('/checkpoint/mitchellw/experiments/open_clip'):
        args.logs = '/checkpoint/mitchellw/experiments/open_clip'
    args.name = name
    args.job_dir = name
    main_with_args(args)


    args = parse_args()

    args.model = 'ViT-B/32'
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
            print('setting default', name, val)

    args.ngpus = 8
    args.batch_size = 128
    args.nodes = 4
    args.lr = 1e-3

    args.partition = 'learnlab'
    args.use_volta32 = False

    args.imagenet_val = '/datasets01/imagenet_full_size/061417/val'
    args.train_data = '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'
    args.train_num_samples = 40000000
    args.dataset_type = 'webdataset'
    
    args.precision = 'amp'
    args.workers = 6
    
    args.epochs = 160
    args.report_to = 'wandb'
    args.seed = 1
    args.ddp_static_graph = True
    args.local_loss = True
    args.dataset_resampled = True
    args.gather_with_grad = True
    args.grad_checkpointing = True
    args.save_frequency = 1
    args.zeroshot_frequency = 1
    args.warmup = 10000

    name = f'benchmark2-{args.nodes}-b32-400m-opt-{args.lr}-{args.beta1}-{args.beta2}-{args.eps}-bs-{args.batch_size * args.ngpus * args.nodes}-{args.precision}-s{args.seed}'
    if os.path.exists('/checkpoint/mitchellw/experiments/open_clip'):
        args.logs = '/checkpoint/mitchellw/experiments/open_clip'
    args.name = name
    args.job_dir = name
    main_with_args(args)


    args = parse_args()

    args.model = 'ViT-B/32'
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
            print('setting default', name, val)

    args.ngpus = 8
    args.batch_size = 512
    args.nodes = 1
    args.lr = 1e-3

    args.partition = 'learnlab'
    args.use_volta32 = False

    args.imagenet_val = '/datasets01/imagenet_full_size/061417/val'
    args.train_data = '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'
    args.train_num_samples = 40000000
    args.dataset_type = 'webdataset'
    
    args.precision = 'amp'
    args.workers = 6
    
    args.epochs = 160
    args.report_to = 'wandb'
    args.seed = 1
    args.ddp_static_graph = True
    args.local_loss = True
    args.dataset_resampled = True
    args.gather_with_grad = True
    args.grad_checkpointing = True
    args.save_frequency = 1
    args.zeroshot_frequency = 1
    args.warmup = 10000

    name = f'benchmark2-{args.nodes}-b32-400m-opt-{args.lr}-{args.beta1}-{args.beta2}-{args.eps}-bs-{args.batch_size * args.ngpus * args.nodes}-{args.precision}-s{args.seed}'
    if os.path.exists('/checkpoint/mitchellw/experiments/open_clip'):
        args.logs = '/checkpoint/mitchellw/experiments/open_clip'
    args.name = name
    args.job_dir = name
    main_with_args(args)

    args = parse_args()

    args.model = 'ViT-B/32'
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
            print('setting default', name, val)

    args.ngpus = 8
    args.batch_size = 512
    args.nodes = 4
    args.lr = 1e-3

    args.partition = 'learnlab'
    args.use_volta32 = False

    args.imagenet_val = '/datasets01/imagenet_full_size/061417/val'
    args.train_data = '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'
    args.train_num_samples = 40000000
    args.dataset_type = 'webdataset'
    
    args.precision = 'amp'
    args.workers = 6
    
    args.epochs = 160
    args.report_to = 'wandb'
    args.seed = 1
    args.ddp_static_graph = True
    args.local_loss = True
    args.dataset_resampled = True
    args.gather_with_grad = True
    args.grad_checkpointing = True
    args.save_frequency = 1
    args.zeroshot_frequency = 1
    args.warmup = 10000

    name = f'benchmark2-{args.nodes}-b32-400m-opt-{args.lr}-{args.beta1}-{args.beta2}-{args.eps}-bs-{args.batch_size * args.ngpus * args.nodes}-{args.precision}-s{args.seed}'
    if os.path.exists('/checkpoint/mitchellw/experiments/open_clip'):
        args.logs = '/checkpoint/mitchellw/experiments/open_clip'
    args.name = name
    args.job_dir = name
    main_with_args(args)

    args = parse_args()

    args.model = 'ViT-B/32'
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
            print('setting default', name, val)

    args.ngpus = 8
    args.batch_size = 256
    args.nodes = 4
    args.lr = 1e-3

    args.partition = 'learnlab'
    args.use_volta32 = False

    args.imagenet_val = '/datasets01/imagenet_full_size/061417/val'
    args.train_data = '/datasets01/laion400m/laion400m-met-release/laion400m-dataset/{00000..41627}.tar'
    args.train_num_samples = 40000000
    args.dataset_type = 'webdataset'
    
    args.precision = 'amp'
    args.workers = 6
    
    args.epochs = 160
    args.report_to = 'wandb'
    args.seed = 1
    args.ddp_static_graph = True
    args.local_loss = True
    args.dataset_resampled = True
    args.gather_with_grad = True
    args.grad_checkpointing = True
    args.save_frequency = 1
    args.zeroshot_frequency = 1
    args.warmup = 10000

    name = f'benchmark2-{args.nodes}-b32-400m-opt-{args.lr}-{args.beta1}-{args.beta2}-{args.eps}-bs-{args.batch_size * args.ngpus * args.nodes}-{args.precision}-s{args.seed}'
    if os.path.exists('/checkpoint/mitchellw/experiments/open_clip'):
        args.logs = '/checkpoint/mitchellw/experiments/open_clip'
    args.name = name
    args.job_dir = name
    main_with_args(args)