from __future__ import print_function
import numpy as np
import pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
import sugar_plus_plus as spp
import matplotlib.gridspec as gridspec
import copy

class hubble_diagram(object):

    def __init__(self, mb, data, cov, zcmb, dmz=None, 
                 host_prop=None, p_host=None, 
                 host_prop_err_minus=None, host_prop_err_sup=None):

        self.mb = mb 
        self.data = data
        self.cov = cov
        self.zcmb = zcmb

        if dmz is None:
            self.dmz = np.zeros_like(self.mb)
        else:
            self.dmz = dmz

        self.host_prop = host_prop
        self.p_host = p_host
        self.host_prop_err_minus = host_prop_err_minus
        self.host_prop_err_sup = host_prop_err_sup
        
        self.ntheta = len(self.data[0]) + 1        
        if self.p_host is not None:
            self.ntheta += 1
        self.theta = np.ones(self.ntheta)
        self.nsn = len(self.mb)
        self.dof = self.nsn - self.ntheta
        self.sigma_int = 0.0
        self.distance_modulus = spp.distance_modulus(self.zcmb)

    def comp_chi2(self, theta):

        residuals = copy.deepcopy(self.mb) - self.distance_modulus

        for i in range(len(self.data[0])):
            residuals -= self.data[:,i] * theta[i]

        if self.p_host is not None:
           residuals -= self.p_host * theta[-2]

        residuals -= theta[-1]

        var = np.zeros_like(self.mb)

        vec_theta = [1.]
        for i in range(len(self.data[0])):
            vec_theta.append(theta[i])
        vec_theta = np.array(vec_theta)

        for sn in range(self.nsn):            
            var[sn] = vec_theta.dot(self.cov[sn].dot(vec_theta.reshape((len(vec_theta),1)))) + self.sigma_int**2 + self.dmz[sn]**2 

        self.residuals = residuals
        self.var = var
        self.chi2 = np.sum(self.residuals**2 / var)

        return self.chi2

    def chi2_dof(self, sigma_int):
        self.sigma_int = sigma_int
        chi2 = self.comp_chi2(self.theta)
        return (chi2/self.dof) - 1             

    def minimize(self):
        # first round for minimizing
        p0 = np.zeros(self.ntheta)
        p0[-1] = -19.2
        
        self.theta = optimize.fmin(self.comp_chi2, p0)
        c = self.chi2_dof(self.sigma_int)
        print('premier c', c) 
        count = 0  
        
        while c > 1e-2:
            results = optimize.fsolve(self.chi2_dof, 0.1)
            self.sigma_int = results[0]
            self.theta = optimize.fmin(self.comp_chi2, self.theta)
            c = self.chi2_dof(self.sigma_int)
            count += 1
            if count > 10:
                break

        print(c)        
        print(self.sigma_int)
        print(self.theta)

    def plot_results(self, param_name):

        self.minimize()
        mu = copy.deepcopy(self.mb)
        mu += - self.theta[-1]
        mu_ajuste = copy.deepcopy(mu)
            
                
        for k in range(len(self.data[0])):            
            mu_ajuste -= self.theta[k]*self.data[:,k]
            
            mb_reduit_k = copy.deepcopy(self.mb) - spp.distance_modulus(self.zcmb)
            mb_reduit_k -= self.theta[-2]*self.p_host
            
            for i in range(len(self.data[0])):
                if i!=k :
                    mb_reduit_k -= self.theta[i]*self.data[:,i]
                    
            plt.figure()
            plt.scatter(self.data[:,k], mb_reduit_k, color='r', marker ='o', s = 20,linewidth=1)
            
            fit_mb_k = self.theta[k]*self.data[:,k] + self.theta[-1]
            plt.plot(self.data[:,k],fit_mb_k,color='b')
            
            plt.title('$m_B$ as a function of parameter %s'%(param_name[k]))
            plt.errorbar(self.data[:,k], mb_reduit_k, linestyle='',
			            yerr=np.sqrt(self.var), xerr=None,
			            ecolor='grey', 
			            alpha=0.4, marker='.', 
			            zorder=0)
            plt.grid()
            plt.xlabel('Parameter %s'%(param_name[k]), fontsize = 12)
            plt.ylabel('$m_B - 5 \t \log_{10} d_l$', fontsize = 12)
            plt.legend(('fit $q_%i \t $%s$ + M_B$ '%(k,param_name[k]), 'Ajusted $m_B$'))
            plt.show()
            
            if self.p_host is not None:
                plt.scatter(self.host_prop, self.data[:,k], c=self.p_host, cmap = 'bwr', marker ='o', s = 20,linewidth=1) 
                plt.title('Parameter %s as a function of host galaxy mass'%(param_name[k]))
                plt.grid()
                plt.xlabel('$\log_{10} \\frac{M_{galaxy}}{M_{\odot}}$', fontsize=12)
                plt.ylabel('Parameter %s'%(param_name[k]), fontsize = 12)
                cb = plt.colorbar()
                cb.set_label('Probability of having this mass')
                plt.show()

        z_span = np.linspace(1E-2,0.15,100)
        mu_th = spp.distance_modulus(z_span)
    
        gs=gridspec.GridSpec(2,1,height_ratios=[2,1])
    
        plt.subplot(gs[0])
        plt.title('Hubble Diagram')
        plt.scatter(self.zcmb,mu, marker ='+', s=20, linewidth=1, color='r')
        plt.scatter(self.zcmb,mu_ajuste, marker ='+', s = 20, linewidth=1,  color = 'b')
        plt.plot(z_span,mu_th, linewidth=1, color = 'g')
        plt.errorbar(self.zcmb, mu_ajuste, linestyle='',
			        yerr=np.sqrt(self.var), xerr=None,
			        ecolor='grey', 
			        alpha=0.4, marker='.', 
			        zorder=0)
        plt.xscale('log')
        plt.xlim([1E-2,0.15])
        plt.legend(('Theoric $\mu$','$\mu =  m_B - M_B$',
                    '$\mu = m_B - M_B + \\alpha \t X_1 -\\beta \t C  -\Delta M \t p$'))
        plt.xticks([],[])
        plt.ylabel('Distance modulus $\mu$', fontsize = 12)


        plt.subplot(gs[1])
        weighted_std = spp.weighted_avg_and_std(self.residuals, np.sqrt(self.var))
        plt.scatter(self.zcmb, self.residuals, 
                    marker ='+', s = 20, 
                    linewidth=1, color = 'b', label = 'wRMS = %.3f mag'%(weighted_std))
        plt.errorbar(self.zcmb, self.residuals, linestyle='',
			        yerr=np.sqrt(self.var), xerr=None,
			        ecolor='grey', 
			        alpha=0.4, marker='.', 
			        zorder=0)
        plt.axhline(linewidth=1, color='g')
        plt.xscale('log')  
        plt.legend()
        plt.xlim([1E-2,0.15])
        plt.xlabel('Redshift', fontsize = 12)
        plt.ylabel('Residuals $\Delta \mu$', fontsize = 12)
    

        plt.subplots_adjust(hspace=0)
        plt.show()

        if self.p_host is not None:

            gs=gridspec.GridSpec(1,2, width_ratios=[1,1])
            
            mu_no_correction = self.residuals + self.theta[-2]*self.p_host[0] 


            plt.subplot(gs[0])
            plt.scatter(self.host_prop, self.residuals, c=self.p_host, cmap = 'bwr', marker ='o', s = 20,linewidth=1)
            plt.title('Hubble diagram residuals with corrected mass step')
            plt.grid()
            plt.ylim([-0.6, 0.4])
            plt.xlabel('$\log_{10} \\frac{M_{galaxy}}{M_{\odot}}$', fontsize = 12)
            plt.ylabel('$\Delta \mu$', fontsize = 12)
            
            
            plt.subplot(gs[1])
            plt.scatter(self.host_prop, mu_no_correction, c=self.p_host, cmap = 'bwr', marker ='o', s = 20,linewidth=1)
            plt.title('Hubble diagram residuals without corrected mass step')
            plt.grid()
            plt.xlabel('$\log_{10} \\frac{M_{galaxy}}{M_{\odot}}$', fontsize = 12)
            cb = plt.colorbar()
            cb.set_label('Probability of having this mass')
            
            plt.subplots_adjust(wspace=0)
            plt.show()


        
    
if __name__ == "__main__":

    file_host = '../../../../lssfr_paper_full_sntable.csv'
    file_salt2 = '../../../../../sands_companion_dataset/sugar_companion_dataset.pkl'
    file_sugar = '../../../../../sugar_analysis/sugar_analysis/meta_sugar.yaml'

    KEY = 0

    mb, params, cov ,zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up= spp.load_salt2(file_salt2, file_host)

    hd_salt2 = hubble_diagram(mb, params, cov ,zcmb, dmz=None,
                              host_prop=host_prop[KEY], p_host=p_host[KEY],
                              host_prop_err_minus=host_prop_err_down[KEY],
                              host_prop_err_sup=host_prop_err_up[KEY])

    #hd_salt2 = hubble_diagram(mb, params, cov , zcmb, dmz=None,
    #                          host_prop=None, p_host=None,
    #                          host_prop_err_minus=None, host_prop_err_sup=None)
    hd_salt2.minimize()
    #hd_salt2.plot_results(['X1', 'C', 'Probability of having this mass'])

    # mb, params, cov ,zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up= spp.load_sugar_data(sn_data=file_sugar, global_mass=file_host)
    # hd_sugar = hubble_diagram(mb, params, cov ,zcmb, dmz,
    #                           host_prop[KEY], p_host[KEY],
    #                          host_prop_err_down[KEY], host_prop_err_up[KEY])
    # hd_sugar.minimize()
    # hd_sugar.plot_results(['q1', 'q2','q3', 'Av'])
