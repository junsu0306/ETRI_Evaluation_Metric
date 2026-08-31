import torch
import torch.nn as nn
import torchvision.models as models
import netvlad


# encoder = models.mobilenet_v2(pretrained=False)
# encoder  = encoder.features
# layers = list(encoder.children())[:-1]


# for block in layers[:-1]:
#         for p in block.parameters():
#             p.requires_grad = False

# encoder = nn.Sequential(*layers)
# model = nn.Module()
# model.add_module('encoder', encoder)
# net_vlad = netvlad.NetVLAD(num_clusters=64, dim=320)
# model.add_module('pool', net_vlad)

# checkpoint = torch.load("./runs/Jun11_08-43-10_mobilenetv2_netvlad/checkpoints/checkpoint.pth.tar", map_location=lambda storage, loc: storage)

# model.load_state_dict(checkpoint['state_dict'])

# total_params = 0
# zero_params = 0
# trian_params =0

# for p in model.parameters():
#     num = p.numel()
#     total_params += num

#     if p.requires_grad:
#         trian_params += p.numel()


# print(f"Total parameters: {total_params}")
# print(f"Train parameters: {trian_params}")


# print(model.pool)


encoder_dim = 256
encoder = models.resnet18(pretrained=True)
# capture only feature part and remove last pool and fc

layers = list(encoder.children())[:-3]
pruning_block = layers[-1]

for l in layers[:-1]:
    for p in l.parameters():
        p.requires_grad = False

encoder = nn.Sequential(*layers)
model = nn.Module()
model.add_module('encoder', encoder)
net_vlad = netvlad.NetVLAD(num_clusters=64, dim=encoder_dim)
model.add_module('pool', net_vlad)

checkpoint = torch.load("./teacher/teacher_resnet-18_netvlad/checkpoints/checkpoint.pth.tar",
                        map_location=lambda storage, loc: storage)

model.load_state_dict(checkpoint['state_dict'])

total_params = 0
zero_params = 0
trian_params = 0


total_params = sum(p.numel() for p in model.parameters())
trian_params = sum(p.numel()
                   for p in model.parameters() if p.requires_grad == True)


zero_params = sum((p == 0).sum().item() for p in model.parameters())


print(f"Total parameters: {total_params}")
print(f"Train parameters: {trian_params}")
print(f"zero params: {zero_params}")
