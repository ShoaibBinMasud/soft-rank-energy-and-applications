# Multivariate soft rank energy: sample efficiency, consistency, and generative modeling
This repository provides two applications of novel multivariate soft rank energy (sRE) and soft rank mmd (sRMMD). (a) Developing a generative model using sRE and sRMMD as the loss functions to produce MNIST-digits, (b) utilizing sRMMD as the loss in a deep generative model to produce valid knockoffs in order to select statistically significant features.
## Package Dependencies to sRMMMD-based knockoff filter
- python=3.6.5
- numpy=1.14.0
- scipy=1.0.0
- pytorch=0.4.1
- cvxpy=1.0.10
- cvxopt=1.2.0
- pandas=0.23.4
## How to run the code
1. To reproduce the MNIST results: <br>
    - Figure 1(b)- run 'mnist_figures_geneartion.py'<br>
    - Figure 1(a)- use lossType = 'mmd' and run 'mnist_figures_geneartion.py'<br>
    - Figure 1(c)- use lossType = 'mmd' and run 'mnist_figures_geneartion.py'<br>
2. To fully reproduce knockoff figures
To reproduce the figures: for MNIST image: run mnist_figures_generation.py and for knockoff results run knockoff_figures_generation.py
## Demo notebooks
- [Examples/MNIST_generaion.ipynb](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/blob/main/Examples/MNIST_generaion.ipynb) A usage example of sRE  as the loss to generate MNIST digits is available in the form of a Jupyter Notebook.
- [Examples/knockoff_synthetic_settings.ipynb](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/blob/main/Examples/knockoff_synthetic_settings.ipynb) code to generate valid knockoffs using sRMMD.
- [Examples/knockoff_real_data.ipynb](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/blob/main/Examples/knockoff_real_data.ipynb) metabolites selection using sRMMD knockoffs on the real data set available in [dataset/Real dataset](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/tree/main/dataset/Real%20dataset)
- [Examples/visualizing_sRMMD_knockoffs.ipynb](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/blob/main/Examples/visualizing_sRMMD_knockoffs.ipynb) Code to visualize sRMMD knockoffs w.r.t. different $\varepsilon$. The corresponding pickle file can be found in [dataset/generated_srmmd_kncokoffs_vs_epsilon/](https://github.com/ShoaibBinMasud/soft-rank-energy-and-applications/tree/main/dataset/generated_srmmd_kncokoffs_vs_epsilon).
