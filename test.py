import torch
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader
import cv2

from data_lodopab import LoDoPab_test
from discriminator import Discriminator
from generator import Generator
from utils import grad_penalty
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "4, 5, 6, 7"
device_ids = [0, 1, 2, 3]

D = Discriminator(3).cuda() # input channels: (X x X) x Y -> output scalar R
G = Generator(2).cuda()     # input channels:  Z x Y -> output channel X
if len(device_ids) > 1:
    D = torch.nn.DataParallel(D, device_ids = device_ids)
    G = torch.nn.DataParallel(G, device_ids = device_ids)

hyperp = {'batch_size': 1, 'epochs': 2000}

data_train = LoDoPab_test("../ground_truth_train", "../low_dose_simulation")
loader = DataLoader(data_train, batch_size = hyperp['batch_size'],shuffle=False)

D_solver = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=0.0001)
G_solver = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.9), weight_decay=0.0001)

idx = 199
D.load_state_dict(torch.load('./parameters/checkpoint%d/D.pt'%(idx)))
G.load_state_dict(torch.load('./parameters/checkpoint%d/G.pt'%(idx)))
D_solver.load_state_dict(torch.load('./parameters/checkpoint%d/D_solver.pt'%(idx)))
G_solver.load_state_dict(torch.load('./parameters/checkpoint%d/G_solver.pt'%(idx)))

G.eval()
D.eval()
for i, (xi, yi) in enumerate(loader):
    # For LoDoPab, xi, yi is [batch_size, 1, 362, 362]
    xi = F.pad(xi, mode = 'constant', pad = [11, 11, 11, 11, 0, 0, 0, 0])
    yi = F.pad(yi, mode = 'constant', pad = [11, 11, 11, 11, 0, 0, 0, 0])
    # Padding such that dimension is [batch_size, 1, 384, 384]
    cv2.imwrite("eval_results/test_gt.jpg",      torch.squeeze(xi).numpy() * 255.0)
    cv2.imwrite("eval_results/test_lowdose.jpg", torch.squeeze(yi).numpy() * 255.0)


    xi = xi.cuda()
    yi = yi.cuda()
    for j in range(10):
        z = torch.rand((yi.size(0), 1, 384, 384)).cuda() # [batch_size, 1, 384, 384]
        Gz = G( torch.cat((z, yi), dim = 1) ) # G(z, y)  [batch_size, 1, 384, 384]
        cv2.imwrite("eval_results/test_generated_{}.jpg".format(j), torch.squeeze(Gz.cpu()).detach().numpy() * 255.0)


    print("finished!")
    time.sleep(1000000)