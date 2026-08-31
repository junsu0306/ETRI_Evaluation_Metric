import os
import torch
import torch.nn as nn
import torchvision.models as models
import netvlad
from utils.pruning.common import mobilenetv2_L_layer_pruning


encoder = models.mobilenet_v2(pretrained=False)
encoder  = encoder.features
layers = list(encoder.children())[:-1]


for block in layers[:-1]:
        for p in block.parameters():
            p.requires_grad = False

encoder = nn.Sequential(*layers)
model = nn.Module() 
model.add_module('encoder', encoder)
net_vlad = netvlad.NetVLAD(num_clusters=64, dim=320)
model.add_module('pool', net_vlad)


  
total_params = 0
zero_params = 0
trian_params =0

for p in model.parameters():
    num = p.numel()
    total_params += num

    if p.requires_grad:
        trian_params += p.numel()
   
   

print(f"Total parameters: {total_params}")
print(f"Train parameters: {trian_params}")



pth_path = "runs/Jul31_05-23-25_mobilenetv2_netvlad"
resume_ckpt = os.path.join(pth_path, 'checkpoints', 'checkpoint.pth.tar')
checkpoint = torch.load(resume_ckpt, map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
model = model.to("cuda:3")


total_params = sum(p.numel() for p in model.parameters())
trian_params = sum(p.numel() for p in model.parameters() if p.requires_grad == True) 
zero_params = sum((p == 0).sum().item() for p in model.parameters())


print(f"Total parameters: {total_params}")
print(f"Train parameters: {trian_params}")
print(f"zero params: {zero_params}")


pruning_model = mobilenetv2_L_layer_pruning(model, sparsity=0.3)





