# This .py file comptues sRMMD, sRE and MMD loss function
# code to compute MMD adapted from https://github.com/OctoberChang/MMD-GAN/blob/master/mmd.py

import torch
import ghalton
import numpy as np
import torch.nn.functional as F
from src.get_plan import plan

min_var_est = 1e-8

def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)



def loss_func(X, Y, sigma_list, reg = 1, losstype = 'sRMMD',  biased=True,):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if losstype == 'mmd':
        mmd = mix_rbf_mmd2(X, Y, sigma_list, biased=True)
        return torch.sqrt(F.relu(mmd))  
    
    else:
        n, d = X.shape
        sequencer = ghalton.Halton(d)
        halton_points = np.array(sequencer.get(2 * n))
        halton_points = torch.from_numpy(halton_points)
        halton_points = halton_points.to(device)
        data = torch.cat((X, Y), 0).to(device)
        a_i = torch.ones(2 * n, requires_grad = False, device = device)/ (2*n)
        b_j = torch.ones(2 * n, requires_grad = False, device = device)/ (2*n)
        G = plan(a_i, data, b_j, halton_points, p=2, eps= reg)
        row_sum = G.sum(axis = 1)
        scaled_G = G / row_sum[:, np.newaxis]
        R = torch.mm(scaled_G, halton_points)
        Rx, Ry = R[0:n],  R[n:] 
        if losstype == 'sRE':
            sRE = energy_test_statistics(Rx, Ry)
            return torch.sqrt(F.relu(sRE))

        elif losstype== 'sRMMD':
            kernel_sRE = mix_rbf_mmd2(Rx, Ry, sigma_list, biased=True)
            return torch.sqrt(F.relu(kernel_sRE))

def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)
    
def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                      
        diag_Y = torch.diag(K_YY)                      
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             
    K_XY_sums_0 = K_XY.sum(dim=0)                     

    Kt_XX_sum = Kt_XX_sums.sum()                       
    Kt_YY_sum = Kt_YY_sums.sum()                       
    K_XY_sum = K_XY_sums_0.sum()                       

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2

# computing Energy distance 
def pairwise_distance(X, Y):
    x_col = X.unsqueeze(1)
    
    y_lin = Y.unsqueeze(0)
    M = torch.sum((x_col - y_lin)**2 , 2)
    M = torch.sqrt(F.relu(M))
    return M
 
def energy_test_statistics(X, Y):
    n = X.shape[0]
    m = Y.shape[0]
    coefficient = n * m / (n + m)
    xx = pairwise_distance(X + 1e-16, X) # to avoid 'divide by zero error'
    yy = pairwise_distance(Y + 1e-16 , Y)
    xy = pairwise_distance(X, Y)
    return coefficient * ( 2 * torch.mean(xy) - torch.mean(xx) - torch.mean(yy))
