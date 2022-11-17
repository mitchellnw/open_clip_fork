import argparse
import os
from training.params import get_default_params

from run_with_submitit import main_with_args, parse_args
"""
cd /p/project/ccstdl/wortsman1/open_clip_fork/src
conda activate open_clip_cuda11.6
export PYTHONPATH=$PWD

cd /p/project/ccstdl/wortsman1/experiments/open_clip
"""

if __name__ == "__main__":

    args = parse_args()

    args.model = 'ViT-L/14'
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)
            print('setting default', name, val)

    args.ngpus = 4
    args.batch_size = 208
    args.nodes = 2
    args.lr = 1e-3


    args.train_data = '/p/fastdata/mmlaion/laion2B-en/{00000..23295}.tar'
    args.train_num_samples = 1000000 #100000000
    args.dataset_type = 'webdataset'
    
    args.precision = 'amp'
    args.workers = 6
    
    args.epochs = 320
    args.report_to = ''
    args.seed = 1
    args.ddp_static_graph = True
    args.local_loss = True
    args.dataset_resampled = True
    args.gather_with_grad = True
    args.grad_checkpointing = True
    args.save_frequency = 1
    args.zeroshot_frequency = 1
    args.warmup = 10000

    args.cpus_per_task = 12
    args.exclude_mem = True
    args.setup = True
    args.partition = 'booster'
    args.account = 'transfernetx'
    args.use_volta32 = False
    args.shared_folder = '/p/project/ccstdl/wortsman1/experiments/open_clip'
    args.timeout = 30
    args.exclude_dist_url = True
    args.srun_args = True
    name = f'l14-2b-l0-opt-{args.lr}-{args.beta1}-{args.beta2}-{args.eps}-bs-{args.batch_size * args.ngpus * args.nodes}-{args.precision}-v{args.seed}-test6'
    args.logs = args.shared_folder
    args.name = name
    args.job_dir = name
    main_with_args(args)

