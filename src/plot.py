import numpy as np
import pandas as pd
from sklearn import linear_model
from src.utils import sample_Y, select
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
from src.parameters import getTestParameter

def plot(xTest, xTestRankSrmmd, xTestSecond, xTestddlk, xTestmmd, xTestgan, d, distType):
    signal_n = 20
    n_experiments = len(xTest)
    nominal_fdr = 0.1
    alpha = getTestParameter(distType)
    normalize = False
    signal_amplitude_vec = [3, 5, 10,15, 20, 25, 30]
    results_pow = pd.DataFrame(columns=['Amplitude'])
    results_fdp = pd.DataFrame(columns=['Amplitude'])
    for amp_id in range(len(signal_amplitude_vec)):
        POW_srmmd ,POW_mmd, POW_ddlk, POW_second, POW_gan = [],[],[],[],[]
        FDP_srmmd, FDP_mmd, FDP_ddlk, FDP_second, FDP_gan = [],[],[],[],[]
        signal_amplitude = signal_amplitude_vec[amp_id]
        print("Running %d experiments with signal amplitude: %.2f" %(n_experiments,signal_amplitude))
        for i in range(n_experiments):
            X = xTest[i]
            y,theta = sample_Y(X, signal_n=signal_n, signal_a=signal_amplitude)
            xk_srmmd= xTestRankSrmmd[i]
            concat_X = np.concatenate((X, xk_srmmd), axis = 1)
            clf = linear_model.Lasso(alpha = alpha, normalize= normalize, max_iter = 500000)     
            clf.fit(concat_X, y)
            Z_r = clf.coef_
            W_r = np.abs(Z_r[:d]) - np.abs(Z_r[d:])
            selected_m, FDP_r, POW_r = select(W_r, theta, nominal_fdr=nominal_fdr)
            FDP_srmmd.append(FDP_r)
            POW_srmmd.append(POW_r) 
            
    # # # #         #second order
            xk_second =  xTestSecond[i]
            concat_X = np.concatenate((X, xk_second), axis = 1)
            clf.fit(concat_X, y)
            clf = linear_model.Lasso(alpha = alpha, normalize= normalize, max_iter = 500000)
            clf.fit(concat_X, y)
            Z_s = clf.coef_
            W_s = np.abs(Z_s[:d]) - np.abs(Z_s[d:])
            selected_m, FDP_s, POW_s = select(W_s, theta, nominal_fdr=nominal_fdr)
            FDP_second.append(FDP_s)
            POW_second.append(POW_s)
            
    # ### mmm
            xk_mmd =  xTestmmd[i]
            concat_X = np.concatenate((X, xk_mmd), axis = 1)
            clf = linear_model.Lasso(alpha = alpha, normalize= normalize, max_iter = 500000)
            clf.fit(concat_X, y)
            Z_m = clf.coef_
            W_m = np.abs(Z_m[:d]) - np.abs(Z_m[d:])
            selected_m, FDP_m, POW_m = select(W_m, theta, nominal_fdr=nominal_fdr)
            FDP_mmd.append(FDP_m)
            POW_mmd.append(POW_m)
    
    # # ddlk
            xk_ddlk =  xTestddlk[i]
            concat_X = np.concatenate((X, xk_ddlk), axis = 1)
            if distType == 'MultivariateStudentT': alpha_d = 0.3
            elif distType == 'SparseGaussian': alpha_d = 0.05
            else: alpha_d = alpha
            clf = linear_model.Lasso(alpha = alpha_d, normalize= normalize, max_iter = 500000)
            clf.fit(concat_X, y)
            Z_d = clf.coef_
            W_d = np.abs(Z_d[:d]) - np.abs(Z_d[d:])
            selected_d, FDP_d, POW_d = select(W_d, theta, nominal_fdr=nominal_fdr)
            FDP_ddlk.append(FDP_d)
            POW_ddlk.append(POW_d)
    
        # # knockoffGAN
            xk_gan =  xTestgan[i]
            concat_X = np.concatenate((X, xk_gan), axis = 1)
            clf = linear_model.Lasso(alpha = alpha, normalize= normalize, max_iter = 500000)
            clf.fit(concat_X, y)
            Z_g = clf.coef_
            W_g = np.abs(Z_g[:d]) - np.abs(Z_g[d:])
            selected_g, FDP_g, POW_g = select(W_g, theta, nominal_fdr=nominal_fdr)
            FDP_gan.append(FDP_g)
            POW_gan.append(POW_g)
            
        results_pow=  results_pow.append({'Amplitude': signal_amplitude,'srmmd': np.mean(POW_srmmd),'ddlk': np.mean(POW_ddlk),
                                           'second-order': np.mean(POW_second), 'mmd': np.mean(POW_mmd), 'gan': np.mean(POW_gan)}, ignore_index = True)
        results_fdp = results_fdp.append({'Amplitude':signal_amplitude,'rank': np.mean(FDP_srmmd),'ddlk': np.mean(FDP_ddlk),
                                           'second-order': np.mean(FDP_second),'mmd': np.mean(FDP_mmd), 'gan': np.mean(FDP_gan)}, ignore_index = True)
    mpl.rcParams['axes.linewidth'] = 1.2
    w= 2
    m= 4
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.5))
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=.2, wspace=.25)
    ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    
    ax[0].plot(signal_amplitude_vec, results_pow['srmmd'], "-8", markersize=m,linewidth = w, label = '$\mathtt{sRMMD}$')
    ax[0].plot(signal_amplitude_vec, results_pow['mmd'], "-8", markersize=m,linewidth = w, label = '$\mathtt{MMD}$')
    ax[0].plot(signal_amplitude_vec, results_pow['second-order'],"-8", markersize=m,linewidth = w, label ='$\mathtt{Second-order}$')
    ax[0].plot(signal_amplitude_vec, results_pow['ddlk'],"-8", markersize=m,linewidth = w, label ='$\mathtt{DDLK}$')
    ax[0].plot(signal_amplitude_vec, results_pow['gan'],"-8", markersize=m,linewidth = w, label ='$\mathtt{KnockoffGAN}$')
    ax[0].set_ylabel('$\mathtt{Power}$', fontsize = 16)
    
    ax[0].set_xlabel('$\mathtt{Amplitude}$')
    
    ax[1].plot(signal_amplitude_vec, results_fdp['rank'],"-8", markersize=m,linewidth = w, label = '$\mathtt{sRMMD}$')
    ax[1].plot(signal_amplitude_vec, results_fdp['mmd'],"-8", markersize=m,linewidth = w, label = '$\mathtt{MMD}$')
    ax[1].plot(signal_amplitude_vec, results_fdp['second-order'],"-8", markersize=m,linewidth = w, label ='$\mathtt{Second-order}$')
    ax[1].plot(signal_amplitude_vec, results_fdp['ddlk'],"-8", markersize=m, linewidth = w,label ='$\mathtt{DDLK}$')
    ax[1].plot(signal_amplitude_vec, results_fdp['gan'],"-8", markersize=m, linewidth = w,label ='$\mathtt{KnockoffGAN}$')
    ax[1].set_ylabel('$\mathtt{FDR}$', fontsize = 14)
    ax[1].axhline(y=0.1, color='grey', linestyle='--')
    ax[1].set_ylim([-0.015,0.5])
    ax[1].set_xlabel('$\mathtt{Amplitude}$')
    plt.savefig(distType+'.pdf')
