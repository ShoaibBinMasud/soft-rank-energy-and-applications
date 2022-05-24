# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 14:33:02 2022

@author: shoaib
"""

import numpy as np
import scipy.special as sp
from operator import mul
from functools import reduce

def sample_multivariate_normal(mu, Sigma, n=1):
    return (np.linalg.cholesky(Sigma) @ np.random.randn(mu.shape[0], n) +
            mu.reshape(-1, 1)).T

def covariance_AR1(p, rho):
    """
    Construct the covariance matrix of a Gaussian AR(1) process
    """
    assert len(
        rho) > 0, "The list of coupling parameters must have non-zero length"
    assert 0 <= max(
        rho) <= 1, "The coupling parameters must be between 0 and 1"
    assert 0 <= min(
        rho) <= 1, "The coupling parameters must be between 0 and 1"

    # Construct the covariance matrix
    Sigma = np.zeros(shape=(p, p))
    for i in range(p):
        for j in range(i, p):
            Sigma[i][j] = reduce(mul, [rho[l] for l in range(i, j)], 1)
    Sigma = np.triu(Sigma) + np.triu(Sigma).T - np.diag(np.diag(Sigma))
    return Sigma


class GaussianAR1:
    """
    Gaussian AR(1) model
    """

    def __init__(self, p, rho, mu=None):
        """
        Constructor
        :param p      : Number of variables
        :param rho    : A coupling parameter
        :return:
        """
        self.p = p
        self.rho = rho
        self.Sigma = covariance_AR1(self.p, [self.rho] * (self.p - 1))
        if mu is None:
            self.mu = np.zeros((self.p, ))
        else:
            self.mu = np.ones((self.p, )) * mu
         
            # self.mu = np.random.binomial(1,0.5,size=self.p)*mu

    def sample(self, n=1, **args):
        """
        Sample the observations from their marginal distribution
        :param n: The number of observations to be sampled (default 1)
        :return: numpy matrix (n x p)
        """
        return sample_multivariate_normal(self.mu, self.Sigma, n)
        # return np.random.multivariate_normal(self.mu, self.Sigma, n)


class GaussianMixtureAR1:
    """
    Gaussian mixture of AR(1) model
    """

    def __init__(self, p, rho_list, mu_list=None, proportions=None):
        # Dimensions
        self.p = p
        # Number of components
        self.K = len(rho_list)
        # Proportions for each Gaussian
        if (proportions is None):
            self.proportions = [1.0 / self.K] * self.K
        else:
            self.proportions = proportions

        if mu_list is None:
            mu_list = [0 for _ in range(self.K)]

        # Initialize Gaussian distributions
        self.normals = []
        # self.Sigma = np.zeros((self.p, self.p))
        for k in range(self.K):
            rho = rho_list[k]
            self.normals.append(GaussianAR1(self.p, rho, mu=mu_list[k]))
            # self.Sigma += self.normals[k].Sigma / self.K

    def sample(self, n=1, **args):
        """
        Sample the observations from their marginal distribution
        :param n: The number of observations to be sampled (default 1)
        :return: numpy matrix (n x p)
        """
        # Sample vector of mixture IDs
        Z = np.random.choice(self.K, n, replace=True, p=self.proportions)
        # Sample multivariate Gaussians
        X = np.zeros((n, self.p))
        for k in range(self.K):
            k_idx = np.where(Z == k)[0]
            n_idx = len(k_idx)
            X[k_idx, :] = self.normals[k].sample(n_idx)
        return X

def scramble(a, axis=1):
    """
    Return an array with the values of `a` independently shuffled along the
    given axis
    """
    b = np.random.random(a.shape)
    idx = np.argsort(b, axis=axis)
    shuffled = a[np.arange(a.shape[0])[:, None], idx]
    return shuffled

class SparseGaussian:
    """
    Uncorrelated but dependent variables
    """
    def __init__(self, p, m):
        self.p = p
        self.m = m
        # Compute true covariance matrix
        pAA = sp.binom(self.p-1, self.m-1) / sp.binom(self.p, self.m)
        pAB = sp.binom(self.p-2, self.m-2) / sp.binom(self.p, self.m)
        self.Sigma = np.eye(self.p)*pAA + (np.ones((self.p,self.p))-np.eye(self.p)) * pAB
        self.mu = np.zeros((self.p,))

    def sample(self, n=1, **args):
        L = np.repeat(np.reshape(range(self.p),(1,self.p)), n, axis=0)
        L = scramble(L)
        S = (L < self.m).astype("int")
        V = np.random.normal(0,1,(n,1))
        X = S * V
        return X

class MultivariateStudentT:
    """
    Multivariate Student's t distribution
    """
    def __init__(self, p, df, rho):
        assert df > 2, "Degrees of freedom must be > 2"
        self.p = p
        self.df = df
        self.rho = rho
        self.normal = GaussianAR1(p, rho)
        self.Sigma = self.normal.Sigma * self.df/(self.df-2.0)
        self.mu = np.zeros((self.p,))

    def sample(self, n=1, **args):
        '''
        Output:
        Produce M samples of d-dimensional multivariate t distribution
        Input:
        mu = mean (d dimensional numpy array or scalar)
        Sigma = scale matrix (dxd numpy array)
        df = degrees of freedom
        n = # of samples to produce
        '''
        Z = self.normal.sample(n)
        G = np.tile(np.random.gamma(self.df/2.,2./self.df,n),(self.p,1)).T
        return Z/np.sqrt(G)
