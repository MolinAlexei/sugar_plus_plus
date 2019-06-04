import numpy as np
import pylab as plt
import copy
from toolbox import distance_modulus
from scipy import optimize

def generate_mc_data(N=10000, Mb=-19.2, alphas=[-0.15, 3.8], stds=[1., 0.1],
                     mb_err=0.03, stds_err = [0.05, 0.01], step=0.12):

    np.random.seed(13)
    z = np.random.uniform(0.01, 0.1, size = N)
    mu = distance_modulus(z)
    mb = mu + Mb
    data = np.zeros((N, len(alphas)))
    data_cov = np.zeros((N, len(alphas)+1, len(alphas)+1))

    for i in range(len(alphas)):
        data[:,i] = np.random.normal(scale=stds[i], size=N)
        
        mb += data[:,i] * alphas[i]
        data[:,i] += np.random.normal(scale=stds[i]*stds_err[i], size=N)
        data_cov[:,i+1, i+1] = (stds[i]*stds_err[i])**2

    mb += np.random.normal(scale=mb_err, size=N)
    data_cov[:,0,0] = mb_err**2

    if step is None:
        return z, mb, data, data_cov
    else:
        mass = np.random.uniform(5,15, size=N)
        filter_higth_mass = (mass > 10)
        proba = np.zeros(N)
        proba[filter_higth_mass] = 1.
        mb += proba * step
        return z, mb, data, data_cov, mass, proba


def comp_chi2(mb, zcmb, data, data_cov, proba, theta):

        residuals = copy.deepcopy(mb) - distance_modulus(zcmb)
        
        for i in range(len(data[0])):            
            residuals -= data[:,i] * theta[i]
        
        if proba is not None:           
           residuals -= proba* theta[-2]
           

        residuals -= theta[-1]

        var = np.zeros_like(mb)

        vec_theta = [1.]
        for i in range(len(data[0])):
            vec_theta.append(theta[i])
        vec_theta = np.array(vec_theta)
        for sn in range(len(mb)):
            
            var[sn] = np.dot(np.dot(vec_theta, data_cov[sn]), vec_theta.reshape((len(vec_theta),1)))

        chi2 = np.sum(residuals**2 / var)

        return chi2





if __name__ == "__main__":

    z, mb, data, data_cov, mass, proba = generate_mc_data(N=1000, Mb=-19.2, alphas=[-0.15, 3.8], stds=[1, 0.1],
                                             mb_err=0.03, stds_err = [0.05, 0.01], step=0)

    
    theta = [-0.15, 3.8, 0.12, -19.2]
    print(comp_chi2(mb, z, data, data_cov, None, theta))

    
