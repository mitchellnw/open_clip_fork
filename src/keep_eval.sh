while [ 1 ]
do
        for i in `ls -t /fsx/home-mitchellw/experimetns/open_clip/clip-bigG14-pd05-ls1-pinit-160k-2e-3-0.95-amp_bfloat16-v1/checkpoints/*`
        do
                bash eval.sh $i
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