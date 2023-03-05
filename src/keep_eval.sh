while [ 1 ]
do
        #for i in `ls -t /fsx/home-mitchellw/experimetns/open_clip/clip-bigG14-pd05-ls1-pinit-160k-2e-3-0.95-amp_bfloat16-v1/checkpoints/*`
        for i in `ls -t /fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/*`
        do
                bash eval.sh $i
        done
sleep 300
done
