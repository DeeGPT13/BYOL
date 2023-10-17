import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import os
import gc
from torch.utils.data import DataLoader
import torchvision
import dataload_flip as dataload
from BYOL import BYOL, train_sup, loss_fn
#os.environ['CUDA_VISIBLE_DEVICES'] = '7'

batch = 16

os.makedirs('BYOL3D',exist_ok=True)
save_file = 'BYOL3D/Flip.pth'
loss_file = 'BYOL3D/Flip.txt'

epochs=50

device = 'cuda'
model = BYOL([128,64,32],[32,8,32])
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0001)

train_dataset = dataload.VidDataset("train_all.csv",transform=torchvision.transforms.Compose([dataload.VidPathToTensor()]))
train_loader = DataLoader(train_dataset, batch_size = batch, shuffle = True, drop_last=True)
train_sup(n_epochs=epochs,optimizer=optimizer, model=model,loss_fn=loss_fn, train_loader=train_loader, save_file=save_file, loss_file=loss_file)
