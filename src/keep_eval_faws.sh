while [ 1 ]
do
        for i in `ls -t /fsx-labs/mitchellw/experiments/openclip2/clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-rs-v1/checkpoints/*`
        do
                bash eval_faws.sh $i clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-rs-v1
        done
        for i in `ls -t /fsx-labs/mitchellw/experiments/openclip2/clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-pinit-fixcinit-rs-v1/checkpoints/*`
        do
                bash eval_faws.sh $i clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-pinit-fixcinit-rs-v1
        done
        for i in `ls -t /fsx-labs/mitchellw/experiments/openclip2/clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-pinit-rs-v1/checkpoints/*`
        do
                bash eval_faws.sh $i clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-pinit-rs-v1
        done
        for i in `ls -t /fsx-labs/mitchellw/experiments/openclip2/clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-fixcinit-rs-v1/checkpoints/*`
        do
                bash eval_faws.sh $i clip-H-14-pd05-bs32k-w8k-opt1e-3-09-095-amp_bfloat16-fixcinit-rs-v1
        done
sleep 300
done

# python -m training.main \
#     --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation \
#     --model ViT-bigG-14 \
#     --precision amp_bfloat16 \
#     --batch-size 200  \
#     --workers 4 \
#     --pretrained /fsx/home-mitchellw/experimetns/open_clip/clip-bigG14-pd05-ls1-pinit-160k-2e-3-0.95-amp_bfloat16-v1/checkpoints/epoch_latest.pt