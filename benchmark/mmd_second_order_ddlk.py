# For DDLK, one needs to install the package from https://github.com/rajesh-lab/ddlk

import numpy as np
from src.gaussian import GaussianKnockoffs
from src.machine import KnockoffGenerator
import argparse
import torch
from benchmark.data_ddlk import get_data
import pytorch_lightning as pl
from ddlk import ddlk, mdn, utils

def mmd_knockoff(xTrain, xTest, distType = 'MultivariateStudentT'):
    n, d = xTrain.shape
    SigmaHat = np.cov(xTrain, rowvar=False)
    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(xTrain,0), method="sdp")
    corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat) 
    if distType == "GaussianAR1": gamma = 1
    if distType == "GaussianMixtureAR1": gamma = 1
    if distType == "MultivariateStudentT": gamma = 1
    if distType == "SparseGaussian": gamma = 1
    pars={"epochs":100, 
          "epoch_length": 20, 
          "d": d,
          "dim_h": int(6*d),
          "batch_size": int(n/4), 
          "lr": 0.01, 
          "lr_milestones": [100],
          "GAMMA":gamma, 
          "losstype": 'mmd',
          "epsilon":None,
          "target_corr": corr_g,
          "sigmas":[1.,2.,4.,8.,16.,32.,64.,128.]
         }
    
    mmd_Machine = KnockoffGenerator(pars)
    mmd_Machine.train(xTrain)
    xTestmmd = [mmd_Machine.generate(xTest[i]) for i in range(len(xTest))]  
    return xTestmmd

## DDLK knockoff
def ddlk_knockoff(xTrain, xTest, distType = 'MultivariateStudentT'):
    trainloader, valloader, testloader = get_data(xTrain)
    pl.trainer.seed_everything(42)
    num_gpus = torch.cuda.device_count()
    gpus = [0] if num_gpus > 0 else None
    
    ((X_mu, ), (X_sigma, )) = utils.get_two_moments(trainloader)
    hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma)
    
    q_joint = mdn.MDNJoint(hparams)
    trainer = pl.Trainer(max_epochs=50, num_sanity_val_steps=1, weights_summary=None, deterministic=True, gpus=gpus)
    trainer.fit(q_joint,train_dataloader= trainloader,val_dataloaders=[valloader])
    
    hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma, reg_entropy=0.01)
    q_knockoff = ddlk.DDLK(hparams, q_joint=q_joint)
    
    trainer = pl.Trainer(max_epochs=50,
                         num_sanity_val_steps=1,
                         deterministic=True,
                         gradient_clip_val=0.5,
                         weights_summary=None, gpus=gpus)
    
    trainer.fit(q_knockoff,train_dataloader=trainloader, val_dataloaders=[valloader])
    
    xTestddlk =[q_knockoff.sample(torch.tensor(xTest[i], dtype=torch.float32)).detach().cpu().numpy() for i in range(len(xTest))]
    return xTestddlk

## Second-order knockoff
def second_kncokoff(xTrain, xTest, distType = 'MultivariateStudentT'):
    SigmaHat = np.cov(xTrain, rowvar=False)
    second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(xTrain,0), method="sdp")
    xTestSecond =  [second_order.generate(xTest[i]) for i in range(len(xTest))]  
    return xTestSecond
