# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 14:33:02 2022

@author: shoaib
"""
import numpy as np
import math
import data


def generateSamples(distType,p):
    '''
    generate a data sampler based on the given distribution
    '''
    if  distType =="SparseGaussian":
        dataSampler = data.SparseGaussian( p, int(0.3*p))
        
    elif distType == "MultivariateStudentT":
        dataSampler = data.MultivariateStudentT(p, 3, 0.5)
        
    elif distType == "GaussianAR1":
        dataSampler = data.GaussianAR1(p, 0.5)
        
    elif distType== "GaussianMixtureAR1":
        k = 4
        prop = (6 + (0.5 + np.arange(k) - k / 2)**2)**0.9
        prop = prop / prop.sum()
        rho_list = [0.6**(i + 0.9) for i in range(k)] #0.6 for V_02, 0.5 for v_03
        mu_list=[20 * i for i in range(k)]
        dataSampler = data.GaussianMixtureAR1(p=p, rho_list=rho_list, mu_list=mu_list, proportions=prop)
        
    return dataSampler


def gen_batches(n_samples, batch_size, n_reps):
    """ Divide input data into batches.
    :param data: input data
    :param batch_size: size of each batch
    :return: data divided into batches
    """
    batches = []
    for rep_id in range(n_reps):
        idx = np.random.permutation(n_samples)
        for i in range(0, math.floor(n_samples/batch_size)*batch_size, batch_size):
            window = np.arange(i,i+batch_size)
            new_batch = idx[window]
            batches += [new_batch]
    return(batches)

def kfilter(W, offset=1.0, q=0.1):
    """
    Adaptive significance threshold with the knockoff filter
    :param W: vector of knockoff statistics
    :param offset: equal to one for strict false discovery rate control
    :param q: nominal false discovery rate
    :return a threshold value for which the estimated FDP is less or equal q
    """
    t = np.insert(np.abs(W[W!=0]),0,0)
    t = np.sort(t)
    ratio = np.zeros(len(t));
    for i in range(len(t)):
        ratio[i] = (offset + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
        
    index = np.where(ratio <= q)[0]
    if len(index)==0:
        thresh = float('inf')
    else:
        thresh = t[index[0]]
       
    return thresh

# generate data Y = X (beta)+ z , z = noise
def sample_Y(X, signal_n=20, signal_a=10.0):
    n,p = X.shape
    beta = np.zeros((p,1))
    beta_nonzero = np.random.choice(p, signal_n, replace=False)
    beta[beta_nonzero,0] = (2*np.random.choice(2,signal_n)-1) * signal_a / np.sqrt(n)
    y = np.dot(X,beta) + np.random.normal(size=(n,1))
    return y,beta


def select(W, beta, nominal_fdr=0.1, offset = 1.0):
    W_threshold = kfilter(W, offset = offset, q=nominal_fdr)
    selected = np.where(W >= W_threshold)[0]
    nonzero = np.where(beta!=0)[0]
    TP = len(np.intersect1d(selected, nonzero))
    FP = len(selected) - TP
    FDP = FP / max(TP+FP,1.0)
    POW = TP / max(len(nonzero),1.0)
    return selected, FDP, POW
    