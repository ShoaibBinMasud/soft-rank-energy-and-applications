import numpy as np
from src.gaussian import GaussianKnockoffs
from src.utils import generateSamples
from src.machine import KnockoffGenerator
from benchmarks import mmd_knockoff, ddlk_knockoff, second_kncokoff
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

# ## DDLK knockoffs
xTestddlk = ddlk_knockoff(xTrain, xTest)

# ## Second-order knockoff
# xTestSecond =  [second_order.generate(xTest[i]) for i in range(len(xTest))] 
xTestSecond = second_kncokoff(xTrain, xTest)

# ## plotting FDR vs POWER tradeoff w.r.t. amplitude
plot(xTest, xTestRankSrmmd, xTestSecond, xTestddlk, xTestmmd, d)
