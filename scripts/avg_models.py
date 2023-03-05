import torch
import os

if __name__ == '__main__':
    
    #model0 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/clip-bigG14-pd05-ls1-pinit-160k-2e-3-0.95-amp_bfloat16-v1/checkpoints/epoch_256.pt', map_location='cpu')
    
    # BEST

    model0 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning/checkpoints/epoch_latest.pt', map_location='cpu')
    model1 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/epoch_latest.pt', map_location='cpu')
    #model2 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-A/checkpoints/epoch_257.pt', map_location='cpu')
    model2 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-A/checkpoints/epoch_261.pt', map_location='cpu')

    model_list = [model0, model1, model2]


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


    model0['state_dict'] = sd#{ k : 1./3 * model0['state_dict'][k] + 1./3 * model1['state_dict'][k] + 1./3 * model2['state_dict'][k] for k in model0['state_dict']}
    model0['epoch'] = 256

    print(model0['epoch'])
    torch.save(model0, 'avg_cp_out.pt')