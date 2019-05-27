

from __future__ import print_function
import numpy as np
import pylab as plt
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from toolbox import distance_modulus
import matplotlib.gridspec as gridspec
import copy

def load_salt2(File):
    #File = os.path.join('data complete/', File)
    dic = pickle.load(open(File))

    X1 = np.array([dic[sn]['salt2_info']['X1'] for sn in dic])
    C = np.array([dic[sn]['salt2_info']['C'] for sn in dic])
    mb = np.array([dic[sn]['salt2_info']['delta_mu'] for sn in dic])
    delta_mu_err = np.array([dic[sn]['salt2_info']['delta_mu_err'] for sn in dic])
    X1_err = np.array([dic[sn]['salt2_info']['X1_err'] for sn in dic])
    C_err = np.array([dic[sn]['salt2_info']['C_err'] for sn in dic])
    delta_mu_X1_cov = np.array([dic[sn]['salt2_info']['delta_mu_X1_cov'] for sn in dic])
    delta_mu_C_cov = np.array([dic[sn]['salt2_info']['delta_mu_C_cov'] for sn in dic])
    X1_C_cov = np.array([dic[sn]['salt2_info']['X1_C_cov'] for sn in dic])
    dmz = np.array([dic[sn]['salt2_info']['dmz'] for sn in dic])
    zcmb = np.array([dic[sn]['salt2_info']['zcmb'] for sn in dic])

    params = np.array([X1,C]).T
    cov = np.zeros((len(X1), 3,3))

    for k in range(len(X1)):
        cov[k] = np.array([[delta_mu_err[k]**2, delta_mu_X1_cov[k],delta_mu_C_cov[k]],
                           [delta_mu_X1_cov[k], X1_err[k]**2,  X1_C_cov[k]],
                           [delta_mu_C_cov[k], X1_C_cov[k], C_err[k]**2]])
    return mb, params, cov ,zcmb, dmz

class hubble_diagram(object):

    def __init__(self, mb, data, cov, zcmb, dmz):

        self.mb = mb 
        self.data = data
        self.cov = cov
        self.zcmb = zcmb
        self.dmz = dmz
       # self.global_mass = global_mass
       # self.lssfr = lssfr
       # self.p_hightmass = p_hightmass
       # self.p_prompt = p_prompt

        self.ntheta = len(self.data[0]) + 1
        self.theta = np.ones(self.ntheta)
        self.nsn = len(self.mb)
        self.dof = self.nsn - self.ntheta
        self.sigma_int = 0.   
    
    def comp_chi2(self, theta):

        residuals = copy.deepcopy(self.mb) - distance_modulus(self.zcmb)
        for i in range(self.ntheta -1):
            residuals -= self.data[:,i] * theta[i]
        residuals -= theta[-1]

        var = np.zeros_like(self.mb)

        vec_theta = [1.]
        for i in range(self.ntheta-1):
            vec_theta.append(theta[i])
        vec_theta = np.array(vec_theta)
        for sn in range(self.nsn):
            
            var[sn] = np.dot(np.dot(vec_theta,self.cov[sn]), vec_theta.reshape((self.ntheta,1))) + 0.03**2+ self.sigma_int**2 + self.dmz[sn]**2

        self.residuals = residuals
        self.var = var
        self.chi2 = np.sum(self.residuals**2 / var)

        return self.chi2


    def chi2_dof(self, sigma_int):
        self.sigma_int = sigma_int
        chi2 = self.comp_chi2(self.theta)
        return (chi2/self.dof) - 1             

    def minimize(self):

        p0 = np.ones(self.ntheta)
        p0[-1] = -19.2
        
        self.theta = optimize.fmin(self.comp_chi2, p0)
        c = (self.comp_chi2(self.theta) / self.dof) - 1
        count = 0
 
        while abs(c)>1E-2:
            print(c)

            self.sigma_int = optimize.fsolve(self.chi2_dof, 0.1)

            self.theta = optimize.fmin(self.comp_chi2, p0)
            c = (self.comp_chi2(self.theta) / self.dof) - 1.
            #if c2 == c:
             #   break
            
            count += 1
            if count > 10:
                break
            

        self.results = c, self.sigma_int, self.theta
        print(c)
        print(self.theta)

    def plot_results(self, param_name):

        self.minimize()
        mu = copy.deepcopy(self.mb)
        mu += - self.theta[-1]
        mu_ajuste = copy.deepcopy(mu)
            
                
        for k in range(self.ntheta-1):
            mu_ajuste -= self.theta[k]*self.data[:,k]
            
            mb_reduit_k = copy.deepcopy(self.mb) - distance_modulus(self.zcmb)
            
            for i in range(self.ntheta-1):
                if i!=k :
                    mb_reduit_k -= self.theta[i]*self.data[:,i]
                
            plt.figure(k)
            plt.scatter(self.data[:,k], mb_reduit_k, color='r', marker ='+', s = 20,linewidth=1)
            
            fit_mb_k=self.theta[k]*self.data[:,k]+self.theta[-1]
            plt.plot(self.data[:,k],fit_mb_k,color='b')
            
            plt.title('$m_B$ as a function of parameter %s'%(param_name[k]))
            plt.grid()
            plt.xlabel('Parameter %s'%(param_name[k]))
            plt.ylabel('$m_B - 5 \t \log_{10} d_l$')
            plt.legend(('fit $q_%i \t $%s$ + M_B$ '%(k,param_name[k]), 'Ajusted $m_B$'))
            plt.show()

          #  if self.global_mass[0] != None:
          #      plt.scatter(self.global_mass, self.data[:,k])
          #      plt.title('Parameter %s as a function of host galaxy mass'%(param_name[k]))
          #      plt.grid()
          #      plt.xlabel('$\log_{10} \f{M_galaxy}{M_solar}$')
          #      plt.ylabel('Parameter %s'%(param_name[k]))
          #      plt.show()


        z_span = np.linspace(1E-2,0.15,100)
        mu_th = distance_modulus(z_span)


    
        gs=gridspec.GridSpec(2,1,height_ratios=[2,1])
    
        plt.subplot(gs[0])
        plt.title('SALT2 Hubble Diagram')
        plt.scatter(self.zcmb,mu, marker ='+', s=20, linewidth=1, color='r')
        plt.scatter(self.zcmb,mu_ajuste, marker ='+', s = 20,linewidth=1,  color = 'b')
        plt.plot(z_span,mu_th, linewidth=1, color = 'g')
        plt.errorbar(self.zcmb, mu_ajuste, linestyle='',
			        yerr=np.sqrt(self.var), xerr=None,
			        ecolor='grey', 
			        alpha=0.9, marker='.', 
			        zorder=0)
        plt.xscale('log')
        plt.xlim([1E-2,0.15])
        plt.legend(('Theoric $\mu$','$\mu =  m_B - M_B$',
                    '$\mu = m_B - M_B + \\alpha \t X_1 -\\beta \t C $'))
        plt.xticks([],[])
        plt.ylabel('Distance modulus $\mu$')


        plt.subplot(gs[1])
        plt.scatter(self.zcmb, self.residuals, 
                    marker ='+', s = 20, 
                    linewidth=1, color = 'b', label = 'RMS = %.3f mag'%(np.std(self.residuals)))
        plt.errorbar(self.zcmb, self.residuals, linestyle='',
			        yerr=np.sqrt(self.var), xerr=None,
			        ecolor='grey', 
			        alpha=0.9, marker='.', 
			        zorder=0)
        plt.axhline(linewidth=1, color='g')
        plt.xscale('log')  
        plt.legend()
        plt.xlim([1E-2,0.15])
        plt.xlabel('Redshift')
        plt.ylabel('Residuals $\Delta \mu$')
    

        plt.subplots_adjust(hspace=0)
        plt.show()

        
    
                   


if __name__ == "__main__":
    #donnees = load_salt2('sugar_companion_dataset.pkl')
    mb, data, cov, zcmb, dmz= load_salt2('data_complete/sugar_companion_dataset.pkl') 
    hd = hubble_diagram(mb, data, cov, zcmb, dmz)
    hd.plot_results(['$X_1$', '$C$'])

    






    
