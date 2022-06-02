# this .py file generate figures in the main body for 'knockoff filters'
# change the distType to 'GaussianAR1' for 2(a), 'GaussianMixtureAR1' for 2(b), 'SparseGaussian' for 2(d)

import numpy as np
from src.gaussian import GaussianKnockoffs
from src.utils import generateSamples
from src.machine import KnockoffGenerator
from benchmark.mmd_second_order_ddlk import mmd_knockoff, ddlk_knockoff, second_kncokoff
from benchmark.knockoffGAN import knockoffgan
from src.plot import plot
from src.parameters import getTrainParameter

n = 2000
d = 100
distType="MultivariateStudentT"
dataSampler = generateSamples(distType, d )
xTrain = dataSampler.sample(n)
SigmaHat = np.cov(xTrain, rowvar=False)
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(xTrain,0), method="sdp")
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
xTest = [dataSampler.sample(n= 200) for i in range(500)]

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
# training 
srmmd_Machine = KnockoffGenerator(pars)
srmmd_Machine.train(xTrain)

# generating knockoffs using several independent test sets (500) 
xTestRankSrmmd = [srmmd_Machine.generate(xTest[i]) for i in range(len(xTest))]  

## BENCHMARKS
## MMD knockoffs
xTestmmd = mmd_knockoff(xTrain, xTest, distType = 'MultivariateStudentT') 

# ## DDLK knockoff
xTestddlk = ddlk_knockoff(xTrain, xTest)

# ## Second-order knockoff
xTestSecond = second_kncokoff(xTrain, xTest, distType = 'MultivariateStudentT')

## knockoffGAN
xTestgan = knockoffgan(xTrain, xTest, distType = 'MultivariateStudentT')

# ## plotting FDR vs POWER tradeoff w.r.t. amplitude
plot(xTest, xTestRankSrmmd, xTestSecond, xTestddlk, xTestmmd, xTestgan, d, distType)
