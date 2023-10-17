import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from math import pi, cos

#os.environ['CUDA_VISIBLE_DEVICES'] = '7'

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(ProjectionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l3 = nn.Sequential(
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class PredictionMLP(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(PredictionMLP, self).__init__()
        self.l1 = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True)
        )
        self.l2 = nn.Linear(mid_dim, out_dim)

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)

        return x


class Backbone(nn.Module):
    
    def __init__(self):
    
        super(Backbone, self).__init__()
        
        self.c1 = nn.Conv3d(3, 16, kernel_size = (1,5,5), stride=1, padding=(0,1,1))
        
        self.c2a = nn.Conv3d(16, 32, kernel_size = (3,3,3), stride=1, padding=(1,1,1))
        self.c2b = nn.Conv3d(32, 32, kernel_size = (3,3,3), stride=1, padding=(1,1,1))
        
        self.c3a = nn.Conv3d(32, 64, kernel_size = (3,3,3), stride=1, padding=(1,1,1))
        self.c3b = nn.Conv3d(64, 64, kernel_size = (3,3,3), stride=1, padding=(1,1,1))
        
        self.c4a = nn.Conv3d(64, 64, kernel_size = (3,3,3), stride=1, padding=(1,1,1))
        self.c4b = nn.Conv3d(64, 64, kernel_size = (3,3,3), stride=1, padding=(1,1,1))
        self.c4c = nn.Conv3d(64, 64, kernel_size = (3,3,3), stride=1, padding=(1,1,1))
        
        self.agg = nn.Conv1d(64,1,1,1)
        
        self.pool = nn.AvgPool3d((1,2,2), stride=(1,2,2))
        self.pool0 = nn.AvgPool2d((2,2), stride=(2,2))
        
        self.glob = nn.AvgPool3d((1,7,7),stride=1)
        self.glob2 = nn.AvgPool3d((1,15,15),stride=1)
        self.relu = nn.ReLU()
        
        self.bn0 = nn.BatchNorm3d(16)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)

    def forward(self, x):
                
        x = self.relu(self.bn0(self.c1(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn1(self.c2a(x)))
        x = self.relu(self.bn1(self.c2b(x)))
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.c3a(x)))
        x = self.relu(self.bn2(self.c3b(x)))
        x1 = self.glob2(x)                   #guided
        x = self.pool(x)
        
        x = self.relu(self.bn2(self.c4a(x))) 
        x2 = self.glob(x)                    #guided
        x = self.relu(self.bn2(self.c4b(x)))
        x3 = self.glob(x)                    #guided
        x = self.relu(self.bn2(self.c4c(x)))
        x = self.glob(x)

        x = torch.squeeze(x)
        x = self.agg(x)
        x = torch.squeeze(x)

        x = x.view(x.size(0), -1)
        
        return x


class BYOL(nn.Module):
    
    def __init__(self,tup1,tup2,base_target_ema=0.996):
    
        super().__init__()        
        # encoder
        self.base_ema = base_target_ema
        features = Backbone()

        # projection MLP
        projection = ProjectionMLP(tup1[0],tup1[1],tup1[2])
        # prediction MLP
        self.online_predictor = PredictionMLP(tup2[0],tup2[1],tup2[2])

        self.online_encoder = nn.Sequential(
            features,
            projection)

        self.target_encoder = copy.deepcopy(self.online_encoder)


    def forward(self,x1,x2):
        
        z1 = self.online_encoder(x1)
        z2 = self.online_encoder(x2)
        
        q1 = self.online_predictor(z1)
        q2 = self.online_predictor(z2)
        
        with torch.no_grad():
            z1_t = self.target_encoder(x1)
            z2_t = self.target_encoder(x2)
       
        loss = loss_fn(q1, q2, z1_t, z2_t)
        
        return loss

    @torch.no_grad()
    def update_moving_average(self, global_step, max_steps):
        
        tau = 1- ((1 - self.base_ema)* (cos(pi*global_step/max_steps)+1)/2) 
        
        for online, target in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            target.data = tau * target.data + (1 - tau) * online.data 

def loss_fn(q1,q2, z1t,z2t):
    
    l1 = - F.cosine_similarity(q1, z1t.detach(), dim=-1).mean()
    l2 = - F.cosine_similarity(q2, z2t.detach(), dim=-1).mean()
    
    return (l1+l2)/2

def train_sup(n_epochs, optimizer, model, loss_fn, train_loader, save_file, loss_file):
    
    l_train=[]

    print('####START####',datetime.datetime.now())
    
    
    for epoch in range(1, n_epochs+1):
        loss_train = 0.0

        model.zero_grad()
        
        counter=0
        for videos in train_loader:

            #videos = torch.cat(videos, dim=0)
            #videos = videos.to(device=device)

            vidA = videos[0]
            vidB = videos[1]

            vidA = vidA.to(device=device)
            vidB = vidB.to(device=device)

            loss = model.forward(vidA,vidB)
            
            #optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            model.update_moving_average(epoch, n_epochs)

            loss_train += loss.item()

            counter +=1

            #if counter%10==0:
                #print(counter,'/',len(train_loader),' ',epoch,datetime.datetime.now())
        
        l_train.append(loss_train/len(train_loader))
        tempa = np.asarray(l_train)
        np.savetxt(loss_file,tempa)
        del tempa

        print('{} Epoch {}, Training loss {}'.format(datetime.datetime.now(), epoch, loss_train/len(train_loader)))
        torch.save(model.state_dict(), save_file)

        if (epoch+1)%5==0:
            sf2 = save_file.split('.')[0]
            sf2 = sf2+'_'+str(epoch)+'.pth'
            torch.save(model.state_dict(), sf2)
