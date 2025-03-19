import torch
checkpoint = torch.load("outputs/epoch=7-step=72500.ckpt")
print(checkpoint.keys())