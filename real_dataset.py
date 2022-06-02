import numpy as np
import pandas as pd
from src.gaussian import GaussianKnockoffs
from src.machine import KnockoffGenerator
from benchmark.mmd_second_order_ddlk import mmd_knockoff, second_kncokoff
from benchmark.knockoffGAN import knockoffgan
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

def missing_value_imputation(data):
    x = data.to_numpy()
    imputer = KNNImputer(n_neighbors = 2, weights="uniform")
    imputed_x =  imputer.fit_transform(x)
    new_data = data.copy()
    new_data.iloc[:, :] = imputed_x
    return new_data

def rf_oob_score(X, X_knockoff, y):
    
    p = X.shape[1]
    clf = RandomForestClassifier(n_estimators = 500, bootstrap = True, oob_score = True, max_features = 'sqrt')
    clf.fit(np.hstack((X, X_knockoff)), y)
    Z = clf.feature_importances_
    W = np.abs(Z[:p]) - np.abs(Z[p : (2 * p)])
    return Z, W

def kfilter(W, offset = 0.0, q = 0.1):
    t = np.insert(np.abs(W[W != 0]), 0, 0) # omitting 0 value and then add zero at the begining
    t = np.sort(t)
    ratio = np.zeros(len(t));
    for i in range(len(t)):
        ratio[i] = (offset + np.sum(W <= -t[i])) / np.maximum(1.0, np.sum(W >= t[i]))
        
    index = np.where(ratio <= q)[0]
    if len(index)== 0:
        thresh = float('inf')
    else:
        thresh = t[index[0]]
       
    return thresh

def selection(W, offset = 1.0, nominal_fdr = 0.1):
    
    W_threshold = kfilter(W, q = nominal_fdr)
    selected = np.where(W >= W_threshold)[0]
    return selected, W_threshold

def metabolites_selection(xTest, test_knockoff, y):
    indices = []
    for i in range(len(xTest)):
        Z, W = rf_oob_score(xTest[i], test_knockoff[i], y)
        S, T = selection(W, offset =  1.0, nominal_fdr =  0.05)# original offset = 1.0
        indices.append(list(S))
        if (i%10)==0: print('no of Simulation completed:', i + 1)
        
    indices_flatten = [j for sub in indices for j in sub]
    selected =  np.unique(indices_flatten)
    count_df = pd.DataFrame()
    unique, counts = np.unique(indices_flatten, return_counts = True)
    count_df['name'] = unique
    count_df['count'] = counts
    count_df = count_df.sort_values(by = ['count'], ascending = False)
    count_df.drop(count_df[count_df['count'] < 1].index, inplace = True)
    selected = count_df['name'].to_numpy()
    return selected



# metabolomics data reading
df =  pd.read_csv('dataset/Real dataset/C18_Negative.txt', sep='\t', dtype = object)
df = df.rename(columns = df.iloc[0])
df = df.drop(df.index[0]).reset_index(drop = True)
df.columns = df.columns.str.replace(' ', '').str.lower()
diagnose = df.columns[1:].str.split('|').str[0]
diagnose= diagnose.str.split(':').str[1]
_, y = np.unique(diagnose, return_inverse=True)
df = df.set_index('factors')
print('unique diagnosis', diagnose.unique(), '\n Metabolite X Samples: ' , df.shape)

# removing metabolites based on missing value percentage
T = [0.8]
data =  df.copy().T
no_of_samples =  data.shape[0]
thresh = int(no_of_samples * T[0])
data = data.dropna(axis = 1, thresh = thresh)#keeping the metabolites which has atleast 70% percenet filled values
missing_percentage_after = data.isnull().sum().sum()/ (data.shape[0] * data.shape[1])
print('After: (Samples X Metabolites): ' , data.shape, '\t\t percentage of missing values: %.3f'
      %missing_percentage_after,'\n')

# missing  value imputation
imputed_data = missing_value_imputation(data)
print('After removing missing values: (Samples X Metabolites): ' , data.shape)

# standardization
standard_data = imputed_data.copy()
standard_data.iloc[:, :] = StandardScaler().fit_transform(standard_data)

xTrain = standard_data.values
n, d = xTrain.shape


print("Generated a training dataset of size: %d x %d." %(xTrain.shape))
SigmaHat = np.cov(xTrain, rowvar=False)
second_order = GaussianKnockoffs(SigmaHat, mu=np.mean(xTrain,0), method="sdp")
corr_g = (np.diag(SigmaHat) - np.diag(second_order.Ds)) / np.diag(SigmaHat)
print('Average absolute pairwise correlation: %.3f.' %(np.mean(np.abs(corr_g))))

pars={"epochs":100, 
      "epoch_length": 20, 
      "d": d,
      "dim_h": int(d*d),
      "batch_size": int(0.5*n), 
      "lr": 0.01, 
      "lr_milestones": [100],
      "GAMMA": 1,
      "losstype": 'sRMMD',
      "epsilon":50,
      "target_corr": corr_g,
      "sigmas":[1.,2.,4.,8.,16.,32.,64.,128.]
     }
print('-- sRMMD---')
srmmd_Machine = KnockoffGenerator(pars)
srmmd_Machine.train(xTrain)

xTest = [xTrain for i in range(100)]

# generating knockoffs using several independent test sets (500) 
xTestRankSrmmd = [srmmd_Machine.generate(xTest[i]) for i in range(len(xTest))]  
selected_r = metabolites_selection(xTest, xTestRankSrmmd, y)
meta_list_r = list(standard_data.columns[selected_r])

## BENCHMARKS
## MMD knockoffs
print('-- MMD---')
xTestmmd = mmd_knockoff(xTrain, xTest)
selected_m = metabolites_selection(xTest, xTestmmd, y)
meta_list_m = list(standard_data.columns[selected_m]) 


# ## Second-order knockoff
print('-- Second-order ---')
xTestSecond = second_kncokoff(xTrain, xTest)
selected_s = metabolites_selection(xTest, xTestSecond, y)
meta_list_s = list(standard_data.columns[selected_s]) 


## knockoffGAN
print('-- KnockoffGAN---')
xTestgan = knockoffgan(xTrain, xTest)
selected_g = metabolites_selection(xTest, xTestgan, y)
meta_list_g = list(standard_data.columns[selected_g]) 


# Final metalist
final_meta_List = pd.DataFrame({'sRMMD': pd.Series(meta_list_r), 'Second-order': pd.Series(meta_list_s),
                                'MMD': pd.Series(meta_list_m), 'knockoffGAN': pd.Series(meta_list_g)})
final_meta_List.to_csv(index=False)

