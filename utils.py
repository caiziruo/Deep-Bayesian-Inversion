import torch
from torch import autograd
import time

def grad_penalty(D, x_Gz1, Gz1_x, Gz1_Gz2, y):
    epsilon = torch.rand(x_Gz1.shape[0], 1, 1, 1).cuda() # [batchsize, 1, 1, 1]

    x_hat_1 = epsilon * x_Gz1 + (1 - epsilon) * Gz1_Gz2 # [batchsize, 2, 384, 384]
    x_hat_2 = epsilon * Gz1_x + (1 - epsilon) * Gz1_Gz2 # [batchsize, 2, 384, 384]

    x_hat_1 = x_hat_1.cuda()
    x_hat_2 = x_hat_2.cuda()
    x_hat_1.requires_grad_()
    x_hat_2.requires_grad_()

    d_hat_1 = D( torch.cat([x_hat_1, y], dim = 1) ) 
    d_hat_2 = D( torch.cat([x_hat_2, y], dim = 1) )

    gradients1 = autograd.grad(outputs = d_hat_1, inputs = x_hat_1,
                              grad_outputs = torch.ones(d_hat_1.size()).cuda(),
                              create_graph = True, retain_graph = True, only_inputs = True)[0]

    gradients2 = autograd.grad(outputs = d_hat_2, inputs = x_hat_2,
                              grad_outputs = torch.ones(d_hat_2.size()).cuda(),
                              create_graph = True, retain_graph = True, only_inputs = True)[0]

    # assert gradients1.shape == (batchsize, 2, 384, 384)
    # assert gradients2.shape == (batchsize, 2, 384, 384)

    gradients = 0.5 * (gradients1 + gradients2)
    norm_gradients = torch.sqrt( torch.sum( gradients ** 2, dim=(1, 2, 3) ) ) # [n, 1]

    # assert norm_gradients.shape == (batchsize,)
    gradient_penalty = torch.mean( (norm_gradients - 1.0) ** 2 )
    return gradient_penalty
