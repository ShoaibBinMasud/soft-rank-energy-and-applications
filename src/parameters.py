# getting the training and test paramerters for sRMMD knockoffs

def getTrainParameter(distType):
    if distType == 'GaussianAR1':
        gamma = 0.01 
        epsilon = 100
    if distType == 'GaussianMixtureAR1':
        gamma = 0.01 
        epsilon = 100
    if distType == 'MultivariateStudentT': 
        gamma = 0.01 
        epsilon = 100
    if distType == 'SparseGaussian':
        gamma = 0.01 
        epsilon = 100
    return gamma, epsilon
    

def getTestParameter(distType):
    if distType == 'GaussianAR1': alpha_r = 0.01
    if distType == 'GaussianMixtureAR1': alpha_r = 0.01
    if distType == 'MultivariateStudentT': alpha_r = 0.01 
    if distType == 'SparseGaussian': alpha_r = 0.06
    return alpha_r

        