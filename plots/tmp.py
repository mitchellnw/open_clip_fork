import torch
import os

if __name__ == '__main__':
    

    model0 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning/checkpoints/epoch_latest.pt', map_location='cpu')
    model1 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-v3/checkpoints/epoch_latest.pt', map_location='cpu')
    model2 = torch.load('/fsx/home-mitchellw/experimetns/open_clip/bigG14-unmasked-tuning-A/checkpoints/epoch_261.pt', map_location='cpu')

    model_list = [model0, model1, model2]

    k = len(model_list)
    kinv = 1./k
    sd = {k : kinv * model_list[0]['state_dict'][k] for k in model_list[0]['state_dict']}
    for i in range(1, len(model_list)):
        sd = {k : sd[k] + kinv * model_list[i]['state_dict'][k] for k in sd}

    model0['state_dict'] = sd
    torch.save(model0, 'model_avg.pt')