import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader

from data_lodopab import LoDoPab_train
from discriminator import Discriminator
from generator import Generator
from utils import grad_penalty
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '7'

D = Discriminator(3).cuda()# input channels (X x X) x Y -> output scalar R
G = Generator(2).cuda() # input channels:  Z x Y -> output channel X

hyperp = {'batch_size': 32, 'epochs': 2000}

data_train = LoDoPab_train("../ground_truth_train", "../low_dose_simulation")
loader = DataLoader(data_train, batch_size=hyperp['batch_size'],shuffle=True)

D_solver = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=0.0001)
G_solver = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=0.0001)

idx = -1
if idx!=-1:
    D.load_state_dict(torch.load('./parameters/checkpoint%d/D.pt'%(idx)))
    G.load_state_dict(torch.load('./parameters/checkpoint%d/G.pt'%(idx)))
    D_solver.load_state_dict(torch.load('./parameters/checkpoint%d/D_solver.pt'%(idx)))
    G_solver.load_state_dict(torch.load('./parameters/checkpoint%d/G_solver.pt'%(idx)))
    
G.train()
D.train()
for e in range(idx+1,hyperp['epochs']):
    for i, (xi, yi) in enumerate(loader):
        # See Equation (33), Appendix D. in 'Deep Bayesian Inversion'

        # For LoDoPab, xi, yi is [batch_size, 1, 362, 362]
        xi = F.pad(xi, mode = 'constant', pad = [11, 11, 11, 11, 0, 0, 0, 0])
        yi = F.pad(yi, mode = 'constant', pad = [11, 11, 11, 11, 0, 0, 0, 0])
        # Padding such that dimension is [batch_size, 1, 384, 384]
        xi = xi.cuda()
        yi = yi.cuda()
        z1 = torch.rand((yi.size(0), 1, 384, 384)).cuda() # [batch_size, 1, 384, 384]
        z2 = torch.rand((yi.size(0), 1, 384, 384)).cuda() # [batch_size, 1, 384, 384]

        Gz1 = G( torch.cat((z1, yi), dim = 1) ) # G(z1, y)  [batch_size, 1, 384, 384]
        Gz2 = G( torch.cat((z2, yi), dim = 1) ) # G(z2, y)  [batch_size, 1, 384, 384]

        x_Gz1 = torch.cat([xi, Gz1], dim = 1) # (x, G(z1, y)) [batch_size, 2, 384, 384]
        Gz1_x = torch.cat([Gz1, xi], dim = 1) # (G(z1, y), x) [batch_size, 2, 384, 384]
        d_x_Gz_y = 0.5 * ( D( torch.cat([x_Gz1, yi], dim = 1) ) + D( torch.cat([Gz1_x, yi], dim = 1) ) ) # [batch_size, 1]

        Gz1_Gz2 = torch.cat([Gz1, Gz2], dim = 1) # (G(z1, y), G(z2, y)) [batch_size, 2, 384, 384]
        d_Gz1_Gz2_y = D( torch.cat([Gz1_Gz2, yi], dim = 1) ) # [batch_size, 1]

        if (i+1) % 5 != 0:
            d_loss = torch.mean(d_Gz1_Gz2_y) - torch.mean(d_x_Gz_y)
            d_grad = 10 * grad_penalty(D, x_Gz1, Gz1_x, Gz1_Gz2, yi)

            D_x_x_y = D( torch.cat( [xi, xi, yi], dim = 1 ) ) # [batch_size, 1]
            d_drift = 0.001 * torch.mean( torch.pow(D_x_x_y, 2) )
            d_total_loss = d_loss + d_drift + d_grad

            D.zero_grad()
            d_total_loss.backward()
            D_solver.step()
        if (i+1) % 5 == 0:
            g_loss = torch.mean(d_x_Gz_y) - torch.mean(d_Gz1_Gz2_y)

            G.zero_grad()
            g_loss.backward()
            G_solver.step()
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [drift loss: %f][grad loss: %f]"
                % (e, hyperp['epochs'], i, len(loader), d_loss.item(), g_loss.item(), d_drift.item(), d_grad.item())
            )
    if (e+1)%100==0:
        os.mkdir('./parameters/checkpoint%d'%(e))
        torch.save(D.state_dict(),'./parameters/checkpoint%d/D.pt'%(e))
        torch.save(G.state_dict(),'./parameters/checkpoint%d/G.pt'%(e))
        torch.save(D_solver.state_dict(),'./parameters/checkpoint%d/D_solver.pt'%(e))
        torch.save(G_solver.state_dict(),'./parameters/checkpoint%d/G_solver.pt'%(e))
