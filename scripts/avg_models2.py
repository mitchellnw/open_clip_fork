import torch
import os

if __name__ == '__main__':
    
    #model0 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/clip-bigG14-pd05-ls1-pinit-160k-2e-3-0.95-amp_bfloat16-v1/checkpoints/epoch_256.pt', map_location='cpu')
    
    # BEST

    model0 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning/checkpoints/epoch_latest.pt', map_location='cpu')
    #model1 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/epoch_287.pt', map_location='cpu')
    model1 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/epoch_latest.pt', map_location='cpu')
    #model2 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-A/checkpoints/epoch_259.pt', map_location='cpu')
    #model2 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-A/checkpoints/epoch_261.pt', map_location='cpu')
    model2 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-A/checkpoints/epoch_latest.pt', map_location='cpu')

    model_list = [model0, model1, model2]

    # # best is epoch_261.pt
    # model0 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/epoch_latest.pt', map_location='cpu')
    # model2 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-A/checkpoints/epoch_261.pt', map_location='cpu')

    #model_list = [model0, model2]

    #model0 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning/checkpoints/epoch_272.pt', map_location='cpu')
    #model0 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning/checkpoints/epoch_271.pt', map_location='cpu')
    #model02 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning/checkpoints/epoch_270.pt', map_location='cpu')
    #model1 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/epoch_287.pt', map_location='cpu')
    #model11 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/epoch_288.pt', map_location='cpu')
    #model12 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/epoch_286.pt', map_location='cpu')

    #model_list = [model0, model1, model02]#, model01, model11, model02, model12]


    k = len(model_list)
    kinv = 1./k
    sd = {k : kinv * model_list[0]['state_dict'][k] for k in model_list[0]['state_dict']}

    for i in range(1, len(model_list)):
        sd = {k : sd[k] + kinv * model_list[i]['state_dict'][k] for k in sd}

    
    todel = []
    #import pdb; pdb.set_trace()
    for k in sd:
        kw,kb = None, None
        if 'ls_1.gamma' in k:
            kw = k.replace('ls_1.gamma', 'attn.out_proj.weight')
            kb = k.replace('ls_1.gamma', 'attn.out_proj.bias')
            one = True
        elif 'ls_2.gamma' in k:
            kw = k.replace('ls_2.gamma', 'mlp.c_proj.weight')
            kb = k.replace('ls_2.gamma', 'mlp.c_proj.bias')
            one = False
        if kw is not None:
            #print(k, kw, kb, kw in sd, kb in sd)
            #import pdb; pdb.set_trace()
            sd[kw] = sd[kw] * sd[k].unsqueeze(1)
            sd[kb] = sd[kb] * sd[k]
            todel.append(k)

    for k in todel:
        del sd[k]


    #import pdb; pdb.set_trace()

    model0['state_dict'] = sd#{ k : 1./3 * model0['state_dict'][k] + 1./3 * model1['state_dict'][k] + 1./3 * model2['state_dict'][k] for k in model0['state_dict']}
    torch.save(model0, 'avg_cp_out2.pt')
    torch.save(sd, 'just_sd.pt')




"""

python ../scripts/avg_models.py && torchrun --nproc_per_node 4 -m training.main  --workers 4     --dataset-type webdataset       --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16 --model ViT-bigG-14-ls1 --batch-size 10 --save-most-recent --log-every-n-steps 100000  --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation --pretrained avg_cp_out.pt
python ../scripts/avg_models2.py && torchrun --nproc_per_node 4 -m training.main  --workers 4     --dataset-type webdataset       --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16 --model ViT-bigG-14 --batch-size 10 --save-most-recent --log-every-n-steps 100000  --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation --pretrained avg_cp_out2.pt

torchrun --nproc_per_node 4 -m training.main  --workers 4     --dataset-type webdataset       --train-num-samples 413000000     --local-loss     --gather-with-grad     --grad-checkpointing       --precision amp_bfloat16 --model ViT-bigG-14-ls1 --batch-size 10 --save-most-recent --log-every-n-steps 100000  --imagenet-val /fsx/rom1504/imagenetval/imagenet_validation --pretrained /fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-A/checkpoints/epoch_259.pt


python clip_benchmark/cli.py eval --pretrained_model  /fsx/home-mitchellw/experimetns/open_clip/ViT-B-32-laion5b-lr1e-3-bs90k/checkpoints/epoch_latest.pt \
--dataset benchmark/datasets_multilingual.txt --dataset_root "/fsx/rom1504/CLIP_benchmark/benchmark/clip_benchmark_datasets/{dataset}"  \
--verbose --output "{dataset}_{pretrained}_{model}_{language}_{task}.json"

python clip_benchmark/cli.py eval --model ViT-B-32 --pretrained /fsx/home-mitchellw/experimetns/open_clip/ViT-B-32-laion5b-lr1e-3-bs90k/checkpoints/epoch_latest.pt \
--dataset benchmark/datasets_multilingual.txt --dataset_root "/fsx/rom1504/CLIP_benchmark/benchmark/clip_benchmark_datasets/{dataset}"  \
--verbose --output "{dataset}_{pretrained}_{model}_{language}_{task}.json"


python clip_benchmark/cli.py eval --model ViT-bigG-14 --pretrained /fsx/home-mitchellw/bigG_14_v1.pt \
--dataset imagenet1k --dataset_root "/fsx/rom1504/CLIP_benchmark/benchmark/clip_benchmark_datasets/{dataset}"  \
--verbose --output "{dataset}_{pretrained}_{model}_{language}_{task}.json"

python clip_benchmark/cli.py eval --model ViT-bigG-14 --pretrained /fsx/home-mitchellw/bigG_14_v1.pt \
--dataset imagenet1k --dataset_root "/fsx/rom1504/CLIP_benchmark/benchmark/clip_benchmark_datasets/{dataset}"  \
--cupl --verbose --output "cupl_{dataset}_{pretrained}_{model}_{language}_{task}.json"



python clip_benchmark/cli.py eval --model ViT-bigG-14 --pretrained /fsx/home-mitchellw/bigG_14_v1.pt \
--dataset benchmark/datasets_clf.txt --dataset_root "/fsx/rom1504/CLIP_benchmark/benchmark/clip_benchmark_datasets/{dataset}"  \
--verbose --output "{dataset}_{pretrained}_{model}_{language}_{task}.json"


python clip_benchmark/cli.py eval --model ViT-bigG-14 --pretrained /fsx/home-mitchellw/bigG_14_v1.pt \
--dataset benchmark/datasets_ret.txt --dataset_root "/fsx/home-mitchellw/datasets/{dataset}"  \
--verbose --output "{dataset}_{pretrained}_{model}_{language}_{task}.json"


"""