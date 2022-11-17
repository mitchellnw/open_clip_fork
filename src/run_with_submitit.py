import argparse
import os
import uuid
from pathlib import Path

from training.main import main_with_args as main_train
from training.params import get_args_parser, get_default_params
import submitit


def parse_args():
    parser = get_args_parser()
    parser = argparse.ArgumentParser(parents=[parser])
    parser.add_argument(
        "--ngpus", default=4, type=int, help="Number of gpus to request on each node"
    )
    parser.add_argument(
        "--nodes", default=1, type=int, help="Number of nodes to request"
    )
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--cpus-per-task", default=10, type=int)

    parser.add_argument(
        "--job-dir", default="", type=str, help="Job dir. Leave empty for automatic."
    )
    parser.add_argument(
        "--shared-folder", default="/checkpoint/mitchellw/experiments/open_clip", type=str,
    )
    parser.add_argument(
        "--partition", default="devlab", type=str, help="Partition where to submit"
    )
    parser.add_argument(
        "--account", default="", type=str, help="Slurm account"
    )
    parser.add_argument(
        "--use-volta32", action="store_true", help="Big models? Use this"
    )
    parser.add_argument(
        "--setup", action="store_true",
    )
    parser.add_argument(
        "--exclude-dist-url", action="store_true",
    )
    parser.add_argument(
        "--exclude-mem", action="store_true",
    )
    parser.add_argument(
        "--srun-args", action="store_true",
    )
    parser.add_argument(
        "--comment",
        default="",
        type=str,
        help="Comment to pass to scheduler, e.g. priority message",
    )
    args = parser.parse_args()

    return args

def get_shared_folder(args):
    p = Path(args.shared_folder)
    p.mkdir(exist_ok=True)
    return p


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(args)), exist_ok=True)
    init_file = get_shared_folder(args) / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, argslist):
        self.argslist = argslist
        self.args = None

    def __call__(self):
        from training.main import main_with_args as main_train

        while len(self.argslist) > 0:
            self.args = self.argslist.pop(0)
            print("Running ", self.args)
            self._setup_gpu_args()
            main_train(self.args)

    def checkpoint(self):
        if self.args is not None:
            self.argslist = [self.args] + self.argslist
        if not self.args.exclude_dist_url:
            for i in range(len(self.argslist)):
                self.argslist[i].dist_url = get_init_file(self.args).as_uri()
        print("Requeuing ", self.argslist)
        empty_trainer = type(self)(self.argslist)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(
            str(self.args.output_dir).replace("%j", str(job_env.job_id))
        )
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main_with_args(args_list, run_as_array=False, h1=False):
    if type(args_list) is not list:
        args_list = [args_list]
    
    first_args = args_list[0]
    for i in range(len(args_list)):
        if args_list[i].job_dir == "":
            args_list[i].job_dir = get_shared_folder(first_args) / "%j"
        else:
            args_list[i].job_dir = get_shared_folder(first_args) / args_list[i].job_dir

    executor = submitit.AutoExecutor(
        folder=first_args.job_dir, slurm_max_num_timeout=30
    )

    num_gpus_per_node = first_args.ngpus
    nodes = first_args.nodes
    timeout_min = first_args.timeout

    partition = first_args.partition
    kwargs = {}
    if first_args.use_volta32:
        kwargs["slurm_constraint"] = "volta32gb"
    if first_args.comment:
        kwargs["slurm_comment"] = first_args.comment

    if first_args.setup:
        kwargs['slurm_setup'] = [
            #'export NCCL_DEBUG=INFO',
            #'export NCCL_DEBUG_SUBSYS=ALL',
            'export CUDA_VISIBLE_DEVICES=0,1,2,3',
            'export NCCL_DEBUG=INFO',
            'export NCCL_IB_TIMEOUT=50',
            'export UCX_RC_TIMEOUT=4s',
            'export NCCL_IB_RETRY_CNT=10',
            'export NCCL_ASYNC_ERROR_HANDLING=1',
            'export WANDB_MODE=offline',
            'export MASTER_PORT=12802',
            """master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)""",
            """export MASTER_ADDR=$master_addr"i" """,
            """echo "MASTER_ADDR="$MASTER_ADDR""",
        ]
    else:
        kwargs['slurm_setup'] = ["""export NCCL_DEBUG=INFO"""]
    if not first_args.exclude_mem:
        kwargs['mem_gb'] = 40 * num_gpus_per_node

    if first_args.account != "":
        kwargs['slurm_account'] = first_args.account

    if first_args.srun_args:
        kwargs['slurm_srun_args'] = ["--cpu_bind=none,v", "--accel-bind=gn"]

    executor.update_parameters(
        # mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node,  # one task per GPU
        cpus_per_task=first_args.cpus_per_task,
        nodes=nodes,
        timeout_min=timeout_min,  # max is 60 * 72
        # Below are cluster dependent parameters
        slurm_partition=partition,
        slurm_signal_delay_s=120,
        #slurm_requeue=True,
        **kwargs,
    )

    executor.update_parameters(name="open-clip", slurm_array_parallelism=15)
    for i in range(len(args_list)):
        if not first_args.exclude_dist_url:
            args_list[i].dist_url = get_init_file(first_args).as_uri()
        args_list[i].output_dir = args_list[i].job_dir
        if i >= 1:
            os.makedirs(args_list[i].output_dir, exist_ok=True)

    if run_as_array:
        jobs = []
        with executor.batch():
            for i in range(len(args_list)):
                trainer = Trainer([args_list[i]])
                job = executor.submit(trainer)
                jobs.append(job)

        for job in jobs:
            print("Submitted job_id:", job.job_id)

    else:
        trainer = Trainer(args_list)
        job = executor.submit(trainer)

        print("Submitted job_id:", job.job_id)


def main_without_args():
    args = parse_args()
    main_with_args(args)


if __name__ == "__main__":
    main_without_args()
