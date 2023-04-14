#!/bin/bash
#SBATCH --partition=g40
#SBATCH --job-name=bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 8
#SBATCH --cpus-per-gpu=12
#SBATCH --gres=gpu:8
#SBATCH --output=%x_%j.out
#SBATCH --open-mode=append
#SBATCH --exclusive
#SBATCH --time=4320
# #SBATCH --exclude=ip-26-0-128-[46,48,85,93-94,101,106,111,123,136,142-143,146,168-169,175,183,189,211,215,223,231,244],ip-26-0-129-[0-1,4,6,11,45,48,60,81-82,84-85,94,105,122],ip-26-0-130-[12-13,19,116,127,132,134,147-148,150,163-164,183,193],ip-26-0-131-[4-5,38,51,77,85,89,107-108,111-112,130,143,150-152,168,182-183,188,239-240,244,247],ip-26-0-132-[7,10,21,37,93,98,107,118,130,139,141-142,149,154,184],ip-26-0-133-[67,76,81,89,111,115,126,131-133,140,145,148,151-152,159-160,226,242],ip-26-0-134-[0,26-27,43,52,61,66,76,83,90-91,105,120,134,141,157,201,219,226-227,248,254],ip-26-0-135-[1,4,22,49,55,64,67,110,118,163,173,184,186,190,192-193,204,208,219,242,255],ip-26-0-136-13,ip-26-0-137-[92,94,97,102,115-116,121,124,139,168,175-176,184,196,212,214,240],ip-26-0-138-[3,13,51,62,66,69,71,79,93,101,159,166,171,178,186,188,208,213],ip-26-0-139-[191,200,214,216,218,226,229,235,237,241,246],ip-26-0-141-[140,146,157,161,166,178,217,228,247],ip-26-0-142-[3,13,21,24,29,33,36,38,41,45,49,67,71,103,106,125,144,146,166,184,186,198,204,217,235,237,246,251,254],ip-26-0-143-[30,39,46,53,61,66,111,121,145,164,171,175,180,206,225,230,235,250]
#SBATCH --requeue
#SBATCH --comment laion

# can get up to 320
# 16 * 256 is right
# 14 * 292

export MASTER_PORT=12802

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

cd /admin/home-mitchellw/forks/open_clip_fork/src
export PYTHONPATH="$PYTHONPATH:/admin/home-mitchellw/forks/open_clip_fork/src"

LR=2e-3
BETA2=0.98
MODEL=ViT-B-32
BS=8192
OPT=customadamw

EXP_NAME="$OPT-$MODEL-$BS-$LR-$BETA2-wit-v1"

srun --comment laion --cpu_bind=v --accel-bind=gn python -m training.main \
    --save-frequency 1 \
    --report-to wandb \
    --train-data="s3://s-laion/wit/full_shards/all/attribution_description/{000000..005838}.tar" \
    --train-num-samples 32768000 \
    --dataset-type webdataset \
    --dataset-resampled \
    --warmup 5000 \
    --batch-size=1024 \
    --epochs=5 \
    --lr $LR \
    --beta2 $BETA2 \
    --workers=10 \
    --report-to wandb \
    --name ${EXP_NAME} \
    --logs ~/git/experimetns/opt3 \
    --model $MODEL \
    --seed 0 \
    --ddp-static-graph \
    --local-loss \
    --gather-with-grad \
    --grad-checkpointing \
    --precision amp_bfloat16 \
    --custom-attention vanilla \
    --save-most-recent \
    --advanced-logging \
    --wandb-project-name open_clip_12 \
    --force-patch-dropout 0.5 \
    --resume 'latest' \
    --delete-previous-checkpoint \
    --opt $OPT

# info.
# 99 - 
# 98 - 
# 95 -
# 9 - 
# 8 - 
# 5 - 

# s3://s-laion/wit/full_shards/all/attribution_description/{000000..005838}.tar
# s3://s-laion/shutterstock/full_shards/photo/{000000..058426}.tar
# s3://s-laion/cc12m/shards/{00000..01242}.tar
# s3://s-laion/reddit/shards/{00000..01201}.tar
# s3://s-laion/yfcc15m/shards/shard_{00000..14825}.tar




Traceback (most recent call last):
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 974, in ast_to_ttir
    generator.visit(fn.parse())
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 874, in visit
    return super().visit(node)
  File "~/git/miniconda3/envs/py2/lib/python3.8/ast.py", line 371, in visit
    return visitor(node)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 184, in visit_Module
    ast.NodeVisitor.generic_visit(self, node)
  File "~/git/miniconda3/envs/py2/lib/python3.8/ast.py", line 379, in generic_visit
    self.visit(item)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 874, in visit
    return super().visit(node)
  File "~/git/miniconda3/envs/py2/lib/python3.8/ast.py", line 371, in visit
    return visitor(node)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 253, in visit_FunctionDef
    has_ret = self.visit_compound_statement(node.body)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 148, in visit_compound_statement
    self.last_ret_type = self.visit(stm
    return super().visit(node)
  File "~/git/miniconda3/envs/py2/lib/python3.8/ast.py", line 371, in visit
    return visitor(node)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 351, in visit_BinOp
    rhs = self.visit(node.right)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 874, in visit
    return super().visit(node)
  File "~/git/miniconda3/envs/py2/lib/python3.8/ast.py", line 371, in visit
    return visitor(node)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 351, in visit_BinOp
    rhs = self.visit(node.right)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 874, in visit
    return super().visit(node)
  File "~/git/miniconda3/envs/py2/lib/python3.8/ast.py", line 371, in visit
    return visitor(node)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 355, in visit_BinOp
    return self._apply_binary_method(method_name, lhs, rhs)
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 343, in _apply_binary_method
    return getattr(lhs, method_name)(rhs, _builder=self.builder)
  File "~/git/git/triton/python/triton/language/core.py", line 29, in wrapper
    return fn(*args, **kwargs)
  File "~/git/git/triton/python/triton/language/core.py", line 521, in __mul__
    return semantic.mul(self, other, _builder)
  File "~/git/git/triton/python/triton/language/semantic.py", line 167, in mul
    input, other = binary_op_type_checking_impl(input, other, builder)
  File "~/git/git/triton/python/triton/language/semantic.py", line 115, in binary_op_type_checking_impl
    check_ptr_type_impl(lhs_sca_ty, rhs_sca_ty, allow_lhs_ptr)
  File "~/git/git/triton/python/triton/language/semantic.py", line 95, in check_ptr_type_impl
    raise IncompatibleTypeErrorImpl(type_a, type_b)
triton.language.semantic.IncompatibleTypeErrorImpl: invalid operands of type pointer<fp64> and triton.language.fp32

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "~/git/miniconda3/envs/py2/lib/python3.8/runpy.py", line 194, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "~/git/miniconda3/envs/py2/lib/python3.8/runpy.py", line 87, in _run_code
    exec(code, run_globals)
  File "~/training/main.py", line 868, in <module>
    exec(code, run_globals)                                                                                                                                                                         [2/1872]
  File "~/training/main.py", line 868, in <module>
    main(sys.argv[1:])
  File "~/training/main.py", line 765, in main
    train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer,
  File "~/training/train.py", line 214, in train_one_epoch
    optimizer.step()
  File "~/git/miniconda3/envs/py2/lib/python3.8/site-packages/torch/optim/optimizer.py", line 280, in wrapper
    out = func(*args, **kwargs)
  File "~/git/miniconda3/envs/py2/lib/python3.8/site-packages/torch/utils/_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "~/training/optimizers/adamw.py", line 260, in step
    update_fn(p, grad, exp_avg, exp_avg_sq, lr, wd, beta1hat, beta2hat, self.eps, True)
  File "~/training/optimizers/adamw.py", line 159, in update_fn
    _opt_kernel2[grid](
  File "~/git/git/triton/python/triton/runtime/autotuner.py", line 110, in run
    return self.fn.run(*args, num_warps=config.num_warps, num_stages=config.num_stages, **kwargs, **config.kwargs)
  File "<string>", line 42, in _opt_kernel2
  File "~/git/git/triton/python/triton/compiler/compiler.py", line 455, in compile
    next_module = compile_kernel(module)
  File "~/git/git/triton/python/triton/compiler/compiler.py", line 375, in <lambda>
    lambda src: optimize_ttir(ast_to_ttir(src, signature, configs[0], constants, debug=debug), arch))
  File "~/git/git/triton/python/triton/compiler/code_generator.py", line 983, in ast_to_ttir
    raise CompilationError(fn.src, node, repr(e)) from e
triton.compiler.errors.CompilationError: at 39:22:
    # load, early exit if nan
    grad = tl.load(offset_grad_ptr, mask = mask)
    #if tl.max(tl.libdevice.isnan(grad), 0) == 0:

    p = tl.load(offset_p_ptr, mask = mask)
    exp_avg = tl.load(offset_exp_avg_ptr, mask = mask)
    exp_avg2 = tl.load(offset_exp_avg2_ptr, mask = mask)

    # stepweight decay

    p = p * (1 - lr * wd)
                      ^
IncompatibleTypeErrorImpl('invalid operands of type pointer<fp64> and triton.language.fp32')