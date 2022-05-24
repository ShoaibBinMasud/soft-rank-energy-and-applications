# Multivariate soft rank energy: sample efficiency, consistency, and generative modeling
This repository provides two applications of novel multivariate soft rank energy (sRE) and soft rank mmd (sRMMD) towards developing a generative model. First, we use sRE and sRMMD as the loss functions in a simple generative model architecture to produce MNIST-digits. We then utilize the sRMMD in a deep generative model in order to produce valid knockoffs.
## Package Dependencies
- python=3.6.5
- numpy=1.14.0
- scipy=1.0.0
- pytorch=0.4.1
- cvxpy=1.0.10
- cvxopt=1.2.0
- pandas=0.23.4
## How to run the code
'Kncokff_generator' folder contains all the functions to generate knockoffs. In order to run the code effectively, move these functions to 'Examples' folder after downloding.
## Examples
- [Examples/MNIST_generaion.ipynb](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/blob/main/Examples/MNIST_generaion.ipynb)
