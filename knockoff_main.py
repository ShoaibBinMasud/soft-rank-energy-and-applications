import numpy as np
from src.gaussian import GaussianKnockoffs
from src.utils import generateSamples
from src.machine import KnockoffGenerator
from benchmarks import mmd_knockoff, ddlk_knockoff, second_kncokoff
# import argparse
# import torch
# import data_custom
# import pytorch_lightning as pl
# from ddlk import ddlk, mdn, utils
from plot import plot
from parameters import getTrainParameter

n = 2000
d = 100
distType="MultivariateStudentT"
dataSampler = generateSamples(distType, d )
xTrain = dataSampler.sample(n)
SigmaHat = np.cov(xTrain, rowvar=False)
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(xTrain,0), method="sdp")
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
xTest = [dataSampler.sample(n= 200) for i in range(200)]

## sRMMD knockoff generator'
gamma, epsilon = getTrainParameter(distType)
pars={"epochs":100, 
      "epoch_length": 20, 
      "d": d,
      "dim_h": int(6*d),
      "batch_size": int(n/4), 
      "lr": 0.01, 
      "lr_milestones": [100],
      "GAMMA":gamma, 
      "losstype": 'sRMMD',
      "epsilon":epsilon,
      "target_corr": corr_g,
      "sigmas":[1.,2.,4.,8.,16.,32.,64.,128.]
     }

srmmd_Machine = KnockoffGenerator(pars)
srmmd_Machine.train(xTrain)
xTestRankSrmmd = [srmmd_Machine.generate(xTest[i]) for i in range(len(xTest))]  

## Benchmarks
## MMD knockoffs
xTestmmd = mmd_knockoff(xTrain, xTest) 

## MMD knockoff generator
# pars={"epochs":100, 
#       "epoch_length": 20, 
#       "d": d,
#       "dim_h": int(6*d),
#       "batch_size": int(n/4), 
#       "lr": 0.01, 
#       "lr_milestones": [100],
#       "GAMMA":1, 
#       "losstype": 'MMD',
#       "epsilon":None,
#       "target_corr": corr_g,
#       "sigmas":[1.,2.,4.,8.,16.,32.,64.,128.]
#      }

# mmd_Machine = KnockoffGenerator(pars)
# mmd_Machine.train(xTrain)
# xTestmmd = [srmmd_Machine.generate(xTest[i]) for i in range(len(xTest))]  

# ## DDLK knockoff
xTestddlk = ddlk_knockoff(xTrain, xTest)
# trainloader, valloader, testloader = data_custom.get_data(xTrain)
# pl.trainer.seed_everything(42)
# num_gpus = torch.cuda.device_count()
# gpus = [0] if num_gpus > 0 else None

# ((X_mu, ), (X_sigma, )) = utils.get_two_moments(trainloader)
# hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma)

# q_joint = mdn.MDNJoint(hparams)
# trainer = pl.Trainer(max_epochs=50, num_sanity_val_steps=1, weights_summary=None, deterministic=True, gpus=gpus)
# trainer.fit(q_joint,train_dataloader= trainloader,val_dataloaders=[valloader])

# hparams = argparse.Namespace(X_mu=X_mu, X_sigma=X_sigma, reg_entropy=0.01)
# q_knockoff = ddlk.DDLK(hparams, q_joint=q_joint)

# trainer = pl.Trainer(max_epochs=50,
#                      num_sanity_val_steps=1,
#                      deterministic=True,
#                      gradient_clip_val=0.5,
#                      weights_summary=None, gpus=gpus)

# trainer.fit(q_knockoff,train_dataloader=trainloader, val_dataloaders=[valloader])

# xTestddlk =[q_knockoff.sample(torch.tensor(xTest[i], dtype=torch.float32)).detach().cpu().numpy() for i in range(len(xTest))]

# ## Second-order knockoff
# xTestSecond =  [second_order.generate(xTest[i]) for i in range(len(xTest))] 
xTestSecond = second_kncokoff(xTrain, xTest)

# ## plotting FDR vs POWER tradeoff w.r.t. amplitude

plot(xTest, xTestRankSrmmd, xTestSecond, xTestddlk, xTestmmd, d)
