# This .py is the knockoff generator 
# code partially adapated from https://github.com/msesia/deepknockoffs
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from src.utils import gen_batches
from src.srmmd import loss_func

np.warnings.filterwarnings('ignore')

class Net(nn.Module):
    """ Deep knockoff network
    """
    def __init__(self, d, dim_h, family="continuous"):
        """ Constructor
        :param d: dimensions of data
        :param dim_h: width of the network (~6 layers are fixed)
        :param family: data type, either "continuous" or "binary"
        """
        super(Net, self).__init__()

        self.d = d
        self.dim_h = dim_h
        self.main = nn.Sequential(
            nn.Linear(2*self.d, self.dim_h, bias=False),
            nn.BatchNorm1d(self.dim_h),
            nn.PReLU(),
            nn.Linear(self.dim_h, self.dim_h, bias=False),
            nn.BatchNorm1d(self.dim_h),
            nn.PReLU(),
            nn.Linear(self.dim_h, self.dim_h, bias=False),
            nn.BatchNorm1d(self.dim_h),
            nn.PReLU(),
            nn.Linear(self.dim_h, self.dim_h, bias=False),
            nn.BatchNorm1d(self.dim_h),
            nn.PReLU(),
            nn.Linear(self.dim_h, self.dim_h, bias=False),
            nn.BatchNorm1d(self.dim_h),
            nn.PReLU(),
            nn.Linear(self.dim_h, self.dim_h, bias=False),
            nn.BatchNorm1d(self.dim_h),
            nn.PReLU(),
            nn.Linear(self.dim_h, self.d),
        )

    def forward(self, x, noise):
        """ Sample knockoff copies of the data
        :param x: input data
        :param noise: random noise seed
        :returns the constructed knockoffs
        """
        x_cat = torch.cat((x,noise),1)
        x_cat[:,0::2] = x
        x_cat[:,1::2] = noise
        return self.main(x_cat)

def norm(X, p=2):
    if(p==np.inf):
        return(torch.max(torch.abs(X)))
    else:
        return(torch.norm(X,p))

class KnockoffGenerator:
    """ Deep Knockoff machine
    """
    def __init__(self, pars, checkpoint_name=None, logs_name=None):
        """ Constructor
        :param pars: dictionary containing the following keys
                'p': dimensions of data
                'epochs': number of training epochs
                'epoch_length': number of iterations over the full data per epoch
                'batch_size': batch size
                'lr': learning rate for main training loop
                'lr_milestones': when to decrease learning rate, unused when equals to number of epochs
                'dim_h': width of the network
                'target_corr': target correlation between variables and knockoffs
                'DELTA': decorrelation penalty hyper-parameter
                'GAMMA': penalty for kernel sRE distance
                'sigmas': kernel widths for the kernel sRE measure (uniform weights)
        :param checkpoint_name: location to save the machine
        :param logs_name: location to save the logfile
        """
        # architecture parameters
        self.d = pars['d']
        self.dim_h = pars['dim_h']

        # optimization parameters
        self.epochs = pars['epochs']
        self.epoch_length = pars['epoch_length']
        self.batch_size = pars['batch_size']
        self.lr = pars['lr']
        self.lr_milestones = pars['lr_milestones']

        # loss function parameters
        self.epsilon = pars['epsilon'] 
        self.sigmas = pars['sigmas']
        self.target_corr = torch.from_numpy(pars['target_corr']).float()
        self.GAMMA = pars['GAMMA']
        self.losstype = pars['losstype']
        self.epsilon = pars['epsilon']
        self.noise_std = 1.0        # noise seed
        self.dim_noise = self.d

      
        self.matching_loss = loss_func
        self.matching_param = self.sigmas
        self.lr = self.lr / np.max([self.GAMMA,  1.0]) # Normalize learning rate to avoid numerical issues
        self.resume_epoch = 0

        self.net = Net(self.d, self.dim_h)        # init the network

    def compute_diagnostics(self, X, Xk, noise, test=False):
        """ Evaluates the different components of the loss function
        :param X: input data
        :param Xk: knockoffs of X
        :param noise: allocated tensor that is used to sample the noise seed
        :return diagnostics: a dictionary containing the following keys:
                 'Mean' : distance between the means of X and Xk
                 'Corr-Diag': correlation between X and Xk
                 'Loss': the value of the loss function
                 'Full': discrepancy between (X',Xk') and (Xk'',X'')
                 'Swap': discrepancy between (X',Xk') and (X'',Xk'')_swap(s)
        """
        # Initialize dictionary of diagnostics
        diagnostics = dict()
        diagnostics["Data"] = "train"

        # Center and scale X, Xk
        mX = X - torch.mean(X,0,keepdim=True)
        mXk = Xk - torch.mean(Xk,0,keepdim=True)
        scaleX  = (mX*mX).mean(0,keepdim=True)
        scaleXk = (mXk*mXk).mean(0,keepdim=True)

        # Correlation between X and Xk
        scaleX[scaleX==0] = 1.0   # Prevent division by 0
        scaleXk[scaleXk==0] = 1.0 # Prevent division by 0
        mXs  = mX  / torch.sqrt(scaleX)
        mXks = mXk / torch.sqrt(scaleXk)
        corr = (mXs*mXks).mean()
        diagnostics["Corr-Diag"] = corr.data.cpu().item()

        ##############################
        # Loss function
        ##############################
        _, loss_display, full_swap, partial_swap = self.loss(X[:noise.shape[0]], Xk[:noise.shape[0]], test=False)
        diagnostics["Loss"]  = loss_display.data.cpu().item()
        diagnostics["Full"] = full_swap.data.cpu().item()
        diagnostics["Swap"] = partial_swap.data.cpu().item()

        # Return dictionary of diagnostics
        return diagnostics

    def loss(self, X, Xk, test=False):
        """ Evaluates the loss function
        :param X: input data
        :param Xk: knockoffs of X
        :param test: evaluate the kernel sRE, regardless the value of GAMMA
        :return loss: the value of the effective loss function
                loss_display: a copy of the loss variable that will be used for display
                full_swap: discrepancy between (X',Xk') and (Xk'',X'')
                partial_swap: discrepancy between (X',Xk') and (X'',Xk'')_swap(s)
        """

        # Divide the observations into two disjoint batches
        n = int(X.shape[0]/2)
        X1,Xk1 = X[:n], Xk[:n]
        X2,Xk2 = X[n:(2*n)], Xk[n:(2*n)]

        # Joint variables
        Z1 = torch.cat((X1,Xk1),1)
        Z2 = torch.cat((Xk2,X2),1)
        Z3 = torch.cat((X2,Xk2),1).clone()
        swap_inds = np.where(np.random.binomial(1,0.5,size=self.d))[0]
        Z3[:,swap_inds] = Xk2[:,swap_inds]
        Z3[:,swap_inds+self.d] = X2[:,swap_inds]

        # Compute the discrepancy between (X,Xk) and (Xk,X)
        full_swap = 0.0
        # Compute the discrepancy between (X,Xk) and (X,Xk)_s
        partial_swap = 0.0
        full_swap = self.matching_loss(Z1, Z2, self.matching_param, reg = self.epsilon, losstype = self.losstype)
        partial_swap = self.matching_loss(Z1, Z3, self.matching_param, reg = self.epsilon, losstype = self.losstype)

        # Penalize correlations between variables and knockoffs
        loss_corr = 0.0
        if self.GAMMA>0:
            # Center X and Xk
            mX  = X  - torch.mean(X,0,keepdim=True)
            mXk = Xk - torch.mean(Xk,0,keepdim=True)
            # Correlation between X and Xk
            eps = 1e-3
            scaleX  = mX.pow(2).mean(0,keepdim=True)
            scaleXk = mXk.pow(2).mean(0,keepdim=True)
            mXs  = mX / (eps+torch.sqrt(scaleX))
            mXks = mXk / (eps+torch.sqrt(scaleXk))
            corr_XXk = (mXs*mXks).mean(0)
            loss_corr = (corr_XXk-self.target_corr).pow(2).mean()

        # Combine the loss functions
        loss = full_swap + partial_swap+ self.GAMMA*loss_corr
        loss_display = loss
        return loss, loss_display, full_swap, partial_swap


    def train(self, X_in, resume = False):
       
        """ Fit the machine to the training data
        :param X_in: input data
        :param resume: proceed the training by loading the last checkpoint
        """

        # Divide data into training/test set
        X = torch.from_numpy(X_in).float()

        # used to compute statistics and diagnostics
        self.SigmaHat = np.cov(X,rowvar=False)
        self.SigmaHat = torch.from_numpy(self.SigmaHat).float()
        self.Mask = torch.ones(self.d, self.d) - torch.eye(self.d)

        # allocate a matrix for the noise realization
        noise = torch.zeros(self.batch_size,self.dim_noise)
        use_cuda = torch.cuda.is_available()

        if resume == True:  # load the last checkpoint
            self.load(self.checkpoint_name)
            self.net.train()
        else:  # start learning from scratch
            self.net.train()
            # Define the optimization method
            self.net_optim = optim.SGD(self.net.parameters(), lr = self.lr, momentum=0.9)
            # Define the scheduler
            self.net_sched = optim.lr_scheduler.MultiStepLR(self.net_optim, gamma=0.1,
                                                            milestones=self.lr_milestones)

        # bandwidth parameters of the Gaussian kernel
        self.matching_param = self.sigmas
        # move data to GPU if available
        if use_cuda:
            self.SigmaHat = self.SigmaHat.cuda()
            self.Mask = self.Mask.cuda()
            self.net = self.net.cuda()
            X = X.cuda()
            noise = noise.cuda()
            self.target_corr = self.target_corr.cuda()

        Xk = torch.zeros_like(X)
        self.Sigma_norm = self.SigmaHat.pow(2).sum()
        self.Sigma_norm_cross = (self.Mask*self.SigmaHat).pow(2).sum()

        # Store diagnostics
        diagnostics = pd.DataFrame()

        # main training loop
        for epoch in range(self.resume_epoch, self.epochs):
            self.net.train()
            self.net_sched.step()
            # divide the data into batches
            batches = gen_batches(X.size(0), self.batch_size, self.epoch_length)

            losses = []
            losses_dist_swap = []
            losses_dist_full = []

            for batch in batches:
                X_batch  = X[batch,:]
                self.net_optim.zero_grad()
                # Run the network
                Xk_batch = self.net(X_batch, self.noise_std*noise.normal_())
                # Compute the loss function
                loss, loss_display, full_swap, partial_swap = self.loss(X_batch, Xk_batch)
                # Compute the gradient
                loss.backward()
                del loss
                # Take a gradient step
                self.net_optim.step()

                # Save history
                losses.append(loss_display.data.cpu().item())
#                 if self.GAMMA>0:
                losses_dist_swap.append(partial_swap.data.cpu().item())
                losses_dist_full.append(full_swap.data.cpu().item())

                # Save the knockoffs
                Xk[batch, :] = Xk_batch.data

            ##############################
            # Compute diagnostics
            ##############################

            # Prepare for testing phase
            self.net.eval()

            # Evaluate the diagnostics on the training data, the following
            # function recomputes the loss on the training data
            diagnostics_train = self.compute_diagnostics(X, Xk, noise, test=False)
            diagnostics_train["Loss"] = np.mean(losses)
#             if(self.GAMMA>0 and self.GAMMA>0):
            diagnostics_train["Full"] = np.mean(losses_dist_full)
            diagnostics_train["Swap"] = np.mean(losses_dist_swap)
            diagnostics_train["Epoch"] = epoch
            diagnostics = diagnostics.append(diagnostics_train, ignore_index=True)
            ##############################
            # Print progress
            ##############################

            print("[%4d/%4d], Loss: %.4f" %
                  (epoch + 1, self.epochs, diagnostics_train["Loss"]), end=", ")
            print(self.losstype, ": %.4f" %
                  (diagnostics_train["Full"] + diagnostics_train["Swap"]), end=", ")
            print("Decorr: %.3f" %
                  (diagnostics_train["Corr-Diag"]), end="")
                
            print("")

    def generate(self, X_in):
        """ Generate knockoff copies
        :param X_in: data samples
        :return Xk: knockoff copy per each sample in X
        """

        X = torch.from_numpy(X_in).float()
        self.net = self.net.cpu()
        self.net.eval()

        # Run the network in evaluation mode
        Xk = self.net(X, self.noise_std*torch.randn(X.size(0),self.dim_noise))
        Xk = Xk.data.cpu().numpy()

        return Xk
