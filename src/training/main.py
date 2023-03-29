import glob
import logging
import os
import re
import subprocess
import sys
import random
from datetime import datetime

import bitsandbytes as bnb

import numpy as np
import torch
from torch import optim
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer
from training.data import get_data
from training.distributed import is_master, init_distributed_device, broadcast_object
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import cosine_lr, const_lr, const_lr_cooldown
from training.train import train_one_epoch, evaluate
from training.file_utils import pt_load, check_exists, start_sync_process, remote_sync
from training.optimizers.customadamw import CustomAdamW
from training.optimizers.clipadamw import ClipAdamW
from training.optimizers.stableadamw import StableAdamW
from training.optimizers.momentadamw import MomentAdamW
from training.optimizers.lion import Lion
from training.optimizers.ladamw import LAdamW
from training.optimizers.ladamw2 import LAdamW2
from training.optimizers.skipadamw import SkipAdamW
from training.optimizers.monitoradamw import MonitorAdamW

from training.ema import ModelEmaV2
from training.optimizers.ulion import ULion
from training.optimizers.rlion import RLion

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None

def backward(total_loss, scaler, custom_scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    elif custom_scaler > 0:
        (total_loss * custom_scaler).backward()
    else:
        total_loss.backward()

def main(args):
    args = parse_args(args)

    if args.train_data is not None and args.train_data.startswith('s3'):
        args.train_data = f'pipe:aws s3 cp {args.train_data} -'

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d`-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)
    if args.advanced_logging:
        args.data_path = os.path.join(args.logs, args.name, "data", str(args.rank))
        if is_master(args) and not os.path.exists(args.data_path):
            os.makedirs(args.data_path)
            
    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        pretrained_image=args.pretrained_image,
        image_mean=args.image_mean,
        image_std=args.image_std,
        aug_cfg=args.aug_cfg,
        force_image_drop_path=args.force_image_drop_path,
        force_text_drop_path=args.force_text_drop_path,
        custom_attention=args.custom_attention,
    )
    # change linear layers to bnb
    if args.fp8:
        from .fp8utils import replace_linear
        replace_linear(model, bnb.nn.LinearFP8)
    if args.fp4:
        from .fp8utils import replace_linear
        replace_linear(model, bnb.nn.LinearFP4)
    if args.fp8global:
        from .fp8utils import replace_linear
        replace_linear(model, bnb.nn.LinearFP8Global)
    if args.fp8mix:
        from .fp8utils import replace_linear
        replace_linear(model, bnb.nn.LinearFP8Mixed)
    if args.int8:
        from .fp8utils import replace_linear
        print('Using real Int8')
        replace_linear(model, bnb.nn.Linear8bitLt)#
    if args.int82:
        from .fp8utils import replace_linear
        print('Using real Int8')
        replace_linear(model, bnb.nn.Linear8bitLt2)#
    if args.int8thresh:
        from .fp8utils import replace_linear
        print('Using real Int8 thresh')
        replace_linear(model, bnb.nn.Linear8bitLtThresh)#
    if args.int8sim:
        from .fp8utils import replace_linear
        replace_linear(model, bnb.nn.LinearInt8)#
    if args.int8castsim:
        from .fp8utils import replace_linear
        replace_linear(model, bnb.nn.LinearInt8Cast)#
    if args.int8mix:
        from .fp8utils import replace_linear
        print('Using real Int8, mixed.')
        replace_linear(model, bnb.nn.Linear8bitLtMixed)#
    if args.slint8:
        print('using switchback linear')
        from .fp8utils import replace_linear
        from tkernels.modules import SwitchBackLinear
        replace_linear(model, SwitchBackLinear)#
    if args.sglint8:
        print('using switchback global linear')
        from .fp8utils import replace_linear
        from tkernels.modules import SwitchBackGlobalLinear
        replace_linear(model, SwitchBackGlobalLinear)#
    if args.snew8:
        print('using switchback new linear')
        from .fp8utils import replace_linear
        from tkernels.modules import SwitchBackNewLinear
        replace_linear(model, SwitchBackNewLinear)#

    model_ema_0, model_ema_1, model_ema_2, model_ema_3 = None, None, None, None
    if args.ema:
        # model_ema_0 = ModelEmaV2(model, 0.99, device=device)
        model_ema_1 = ModelEmaV2(model, 0.999, device=device)
        # model_ema_2 = ModelEmaV2(model, 0.9999, device=device)
        # model_ema_3 = ModelEmaV2(model, 0.99999, device=device)
    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        if args.int8 or args.int8sim or args.fp8 or args.int8castsim or args.int82 or args.int8thresh or args.int8mix or args.fp8global or args.fp4 or args.fp8mix or args.slint8 or args.sglint8 or args.snew8:
            model = model.to(device)

        if args.rms_load is not None:
            # from .legacy_ddp import LegacyDistributedDataParallel
            # import torch.distributed as dist
            # groups = [dist.new_group([j]) for j in range(args.world_size)]
            # model = LegacyDistributedDataParallel(model, process_group=groups[args.rank])
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
        include = lambda n, p: not exclude(n, p)

        # """
        # torchrun --nproc_per_node 2 -m training.main   \
        # --batch-size 208   --workers 2 --model ViTls0-B-32     --dataset-type webdataset   \
        # --train-data="pipe:aws s3 cp s3://s-datasets/laion400m/laion400m-dat-release/{00000..41455}.tar -"  \
        # --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing \
        # --precision amp_bfloat16
        # """

        named_parameters = list(model.named_parameters())
        gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
        rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

        if args.opt.lower() == 'customadamw':
            optimizer = CustomAdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif args.opt.lower() == 'customadamw0.99':
            exclude = lambda n, p: (p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n) and ('conv1' not in n)
            include = lambda n, p: not exclude(n, p) and 'conv1' not in n
            named_parameters = list(model.named_parameters())
            gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
            rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
            conv1_params = [p for n, p in named_parameters if 'conv1' in n and p.requires_grad]
            #torch.distributed.barrier()
            print('0.99 custom adamw')
            optimizer = CustomAdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0., 'betas': (args.beta1, 0.99)},
                    {"params": rest_params, "weight_decay": args.wd, 'betas': (args.beta1, 0.99)},
                    {"params": conv1_params, "weight_decay": args.wd, 'betas': (args.beta1, args.beta2)},
                ],
                lr=args.lr,
                eps=args.eps,
                individual_betas=True
            )
        elif args.opt.lower() == 'clipadamw':
            optimizer = ClipAdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif args.opt.lower() == 'skipadamw':
            optimizer = SkipAdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif args.opt.lower() == 'monitoradamw':
            optimizer = MonitorAdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif args.opt.lower() == 'ladamw':
            optimizer = LAdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif args.opt.lower() == 'ladamw2':
            optimizer = LAdamW2(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif args.opt.lower() == 'stableadamw':
            optimizer = StableAdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif args.opt.lower() == 'momentadamw':
            optimizer = MomentAdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        elif args.opt.lower() == 'lion':
            optimizer = Lion(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                #eps=args.eps,
            )
        elif args.opt.lower() == 'ulion':
            optimizer = ULion(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                #eps=args.eps,
            )
        elif args.opt.lower() == 'rlion':
            optimizer = RLion(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                #eps=args.eps,
            )
        elif args.opt.lower() == 'dog':
            print('using dog')
            from dog import DoG
            optimizer = DoG(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
            )
        elif args.opt.lower() == 'ldog':
            print('using ldog')
            from dog import LDoG
            optimizer = LDoG(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
            )
        else:
            optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )
        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = GradScaler() if args.precision == "amp" else None
        optimizer.rank = args.rank
        optimizer.precision = args.precision
        optimizer.custom_scaler = args.custom_scaler

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")

            if args.ema:
                print('resuming EMAs')
                # ema_0_sd = torch.load(args.resume.replace('epoch', f'ema_0'), map_location='cpu')['state_dict']
                # model_ema_0.module.load_state_dict(ema_0_sd)
                ema_1_sd = torch.load(args.resume.replace('epoch', f'ema_1'), map_location='cpu')['state_dict']
                model_ema_1.module.load_state_dict(ema_1_sd)
                # ema_2_sd = torch.load(args.resume.replace('epoch', f'ema_2'), map_location='cpu')['state_dict']
                # model_ema_2.module.load_state_dict(ema_2_sd)
                # ema_3_sd = torch.load(args.resume.replace('epoch', f'ema_3'), map_location='cpu')['state_dict']
                # model_ema_3.module.load_state_dict(ema_3_sd)
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            print('using const lr')
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')


    # FIXME: Only set necessary vars.
    model.apply(lambda m : setattr(m, 'advanced_logging', args.advanced_logging))
    if args.advanced_logging:
        for n, m in model.named_modules():
            setattr(m, 'module_name', n)
        model.apply(lambda m: setattr(m, 'data_path', args.data_path))
        model.apply(lambda m: setattr(m, 'logger_file', None))
        model.apply(lambda m: setattr(m, 'iter', None))
        

    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return

    if args.rms_load is not None:
        from open_clip import ClipLoss, get_cast_dtype
        from .precision import get_autocast
        model.train()
        loss = ClipLoss(
            local_loss=args.local_loss,
            gather_with_grad=args.gather_with_grad,
            cache_labels=True,
            rank=args.rank,
            world_size=args.world_size,
            use_horovod=args.horovod)
        
        rank = args.rank
        rank = 7
        images = torch.load(os.path.join(args.rms_load, f'images_{rank}.pt'))
        texts = torch.load(os.path.join(args.rms_load, f'texts_{rank}.pt'))
        data['train'].set_epoch(5)
        #for batch in data['train'].dataloader:
        optimizer.zero_grad()
        images, texts = next(iter(data['train'].dataloader))

        # rank0 : 1.8
        
        d = images.size(0)
        if args.rank == 0:
            texts = texts[:d//2]            
            images = images[:d//2]
        else:
            texts = texts[d//2:]            
            images = images[d//2:]

        autocast = get_autocast(args.precision)
        cast_dtype = get_cast_dtype(args.precision)
        images = images.to(device=device, dtype=cast_dtype, non_blocking=True)
        texts = texts.to(device=device, non_blocking=True)
        with autocast():
            image_features, text_features, logit_scale = model(images, texts)
            total_loss = loss(image_features, text_features, logit_scale)

        backward(total_loss, scaler, args.custom_scaler)
        optimizer.step()

        #print(f'Loss {args.rank} : {total_loss}')

        for n, p in model.named_parameters():
            if n == 'module.visual.conv1.weight':
                saved_p = p
                out = np.sqrt(optimizer.state[saved_p]['rms_mean'])
        print(f'Results {args.rank} : {out}')

        exit()


    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer,
                        model_ema_0, model_ema_1, model_ema_2, model_ema_3)
        completed_epoch = epoch + 1

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            evaluate(model, data, completed_epoch, args, writer)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            
            if args.ema:
                count = completed_epoch * data['train'].dataloader.num_batches
                print('count is', count)
                # torch.save({k : v.clone().detach() / (1 - (0.99 ** count)) for k, v in model_ema_0.module.state_dict().items()},
                #            os.path.join(args.checkpoint_path, f"dema_0_{completed_epoch}.pt"))
                # torch.save({k : v.clone().detach() / (1 - (0.999 ** count)) for k, v in model_ema_1.module.state_dict().items()},
                #            os.path.join(args.checkpoint_path, f"dema_1_{completed_epoch}.pt"))
                # torch.save({k : v.clone().detach() / (1 - (0.9999 ** count)) for k, v in model_ema_2.module.state_dict().items()},
                #            os.path.join(args.checkpoint_path, f"dema_2_{completed_epoch}.pt"))
                # torch.save({k : v.clone().detach() / (1 - (0.99999 ** count)) for k, v in model_ema_3.module.state_dict().items()},
                #            os.path.join(args.checkpoint_path, f"dema_3_{completed_epoch}.pt"))

                # torch.save({'state_dict' : model_ema_0.module.state_dict(), 'count' : count, 'decay' : 0.99},
                #            os.path.join(args.checkpoint_path, f"ema_0_{completed_epoch}.pt"))
                torch.save({'state_dict' : model_ema_1.module.state_dict(), 'count' : count, 'decay' : 0.999},
                           os.path.join(args.checkpoint_path, f"ema_1_{completed_epoch}.pt"))
                # torch.save({'state_dict' : model_ema_2.module.state_dict(), 'count' : count, 'decay' : 0.9999},
                #            os.path.join(args.checkpoint_path, f"ema_2_{completed_epoch}.pt"))
                # torch.save({'state_dict' : model_ema_3.module.state_dict(), 'count' : count, 'decay' : 0.99999},
                #            os.path.join(args.checkpoint_path, f"ema_3_{completed_epoch}.pt"))
                
                if args.delete_previous_checkpoint:
                    previous_checkpoints = [
                        os.path.join(args.checkpoint_path, f"ema_{ii}_{completed_epoch - 1}.pt") for ii in range(4)
                    ]
                    for previous_checkpoint in previous_checkpoints:
                        if os.path.exists(previous_checkpoint):
                            os.remove(previous_checkpoint)

            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])
