
from __future__ import print_function
import numpy as np
import pylab as plt
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
from scipy import optimize
from toolbox import distance_modulus
from toolbox import weighted_avg_and_std
import yaml

from test_hd import generate_mc_data

import matplotlib.gridspec as gridspec
import copy

def load_salt2(File, File_host):

    table_host = np.loadtxt(File_host, comments='#', delimiter=',', dtype='str',skiprows=1)

    sn_name_host =  table_host[:,16]

    dic = pickle.load(open(File))
    sn_name_dic = dic.keys()

    sn_name = []

    for sn in sn_name_host:
        if sn in sn_name_dic:
            sn_name.append(sn)


    X1 = np.array([dic[sn]['salt2_info']['X1'] for sn in sn_name])
    C = np.array([dic[sn]['salt2_info']['C'] for sn in sn_name])
    mb = np.array([dic[sn]['salt2_info']['delta_mu'] for sn in sn_name])
    delta_mu_err = np.array([dic[sn]['salt2_info']['delta_mu_err'] for sn in sn_name])
    X1_err = np.array([dic[sn]['salt2_info']['X1_err'] for sn in sn_name])
    C_err = np.array([dic[sn]['salt2_info']['C_err'] for sn in sn_name])
    delta_mu_X1_cov = np.array([dic[sn]['salt2_info']['delta_mu_X1_cov'] for sn in sn_name])
    delta_mu_C_cov = np.array([dic[sn]['salt2_info']['delta_mu_C_cov'] for sn in sn_name])
    X1_C_cov = np.array([dic[sn]['salt2_info']['X1_C_cov'] for sn in sn_name])
    dmz = np.array([dic[sn]['salt2_info']['dmz'] for sn in sn_name])
    zcmb = np.array([dic[sn]['salt2_info']['zcmb'] for sn in sn_name])


    cov = np.zeros((len(X1), 3, 3))

    for k in range(len(X1)):
        cov[k] = np.array([[delta_mu_err[k]**2, delta_mu_X1_cov[k], delta_mu_C_cov[k]],
                           [delta_mu_X1_cov[k], X1_err[k]**2,  X1_C_cov[k]],
                           [delta_mu_C_cov[k], X1_C_cov[k], C_err[k]**2]])
    

    global_mass = np.zeros_like(X1)
    gmass_err_down = np.zeros_like(X1)
    gmass_err_up = np.zeros_like(X1)
    lssfr = np.zeros_like(X1)
    lssfr_err_down = np.zeros_like(X1)
    lssfr_err_up = np.zeros_like(X1)
    p_hightmass = np.zeros_like(X1)
    p_prompt = np.zeros_like(X1)

    count=0
    for k,sn in enumerate(sn_name_host):
        if sn in sn_name :
            global_mass[count] = table_host[k,4].astype(float)
            gmass_err_down[count] = table_host[k,5].astype(float)
            gmass_err_up[count] = table_host[k,6].astype(float)
            lssfr[count] = table_host[k,10].astype(float)
            lssfr_err_down[count] = table_host[k,11].astype(float)
            lssfr_err_up[count] = table_host[k,12].astype(float)
            p_hightmass[count] = table_host[k,17].astype(float)
            p_prompt[count] = table_host[k,18].astype(float)
            count += 1

    params = np.array([X1,C]).T

    host_prop = [global_mass, lssfr]
    p_host = [p_hightmass, p_prompt]
    host_prop_err_down = [gmass_err_down, lssfr_err_down]
    host_prop_err_up = [gmass_err_up, lssfr_err_up]

    return mb, params, cov ,zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up

def load_sugar_data(sn_data='meta_sugar.yaml',
                    global_mass='lssfr_paper_full_sntable.csv'):


    host = np.loadtxt(global_mass, skiprows=1, delimiter=',', dtype='str')
    gmass = host[:,4].astype(float)
    gmass_err_down = host[:,5].astype(float)
    gmass_err_up = host[:,6].astype(float)
    lssfr = host[:,10].astype(float)
    lssfr_err_down = host[:,11].astype(float)
    lssfr_err_up = host[:,12].astype(float)
    new_sn_name = host[:,16]
    P_highmass = host[:,17].astype(float)
    P_prompt = host[:,18].astype(float) # (1 - p_delayed I guess)                                                                                                        

    dic = yaml.load(open(sn_data))
    sn_name = dic['sn_name']
    z = dic['z_cmb']
    Filtre = np.array([True]*len(sn_name))
    for i in range(len(sn_name)):
        if z[i]<0.01:
            Filtre[i] = False
        if sn_name[i] not in new_sn_name :
            Filtre[i] = False
        if abs(dic['params_sugar'][i,1])>15:
            Filtre[i] = False
    sn_name = sn_name[Filtre]
    data = dic['params_sugar'][:,1:][Filtre]
    mb = dic['params_sugar'][:,:1][Filtre].T
    mb = mb[0]

    i = 0
    j = 0
    M_host = np.zeros_like(data[:,0])
    Mass_err_down = np.zeros_like(data[:,0])
    Mass_err_up = np.zeros_like(data[:,0])
    LSSFR = np.zeros_like(data[:,0])
    LSSFR_err_down = np.zeros_like(data[:,0])
    LSSFR_err_up = np.zeros_like(data[:,0])
    p_delayed = np.zeros_like(data[:,0])
    p_highmass = np.zeros_like(data[:,0])
    for i in range(len(sn_name)):
        for j in range(len(new_sn_name)):
            if i == j :
                M_host[i] = gmass[j]
                Mass_err_down[i] = gmass_err_down[j]
                Mass_err_up[i] = gmass_err_up[j]
                LSSFR[i] = lssfr[j]
                LSSFR_err_down[i] = lssfr_err_down[j]
                LSSFR_err_up[i] = lssfr_err_up[j]
                p_delayed[i] = 1.-P_prompt[j]
                p_highmass[i] = P_highmass[j]


    zcmb = dic['z_cmb'][Filtre]
    zhelio = dic['z_helio'][Filtre]
    zd_err = [0.001 for k in range(len(zcmb))]
    mz_err = (5*zd_err[0])/(zcmb*np.log(10))

    data_cov = dic['cov_params'][Filtre]
    data_cov[:,0,0] += mz_err**2

    host_prop = [M_host, LSSFR]
    p_host = [p_highmass, p_delayed]
    host_prop_err_down = [Mass_err_down, LSSFR_err_down]
    host_prop_err_up = [Mass_err_up, LSSFR_err_up]

    return mb.T, data, data_cov, zcmb, zd_err, host_prop, p_host, host_prop_err_down, host_prop_err_up

class hubble_diagram(object):

    def __init__(self, mb, data, cov, zcmb, dmz=0, host_prop=None, p_host=None, host_prop_err_minus=None, host_prop_err_sup=None):


        self.mb = mb 
        self.data = data
        self.cov = cov
        self.zcmb = zcmb
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
        self.sigma_int = 0.   
    
    def comp_chi2(self, theta):

        residuals = copy.deepcopy(self.mb) - distance_modulus(self.zcmb)
        
        for i in range(len(self.data[0])):
            residuals -= self.data[:,i] * theta[i]
        if self.p_host is not None:
           residuals -= self.p_host[0] * theta[-2]

        residuals -= theta[-1]

        var = np.zeros_like(self.mb)

        vec_theta = [1.]
        for i in range(len(self.data[0])):
            vec_theta.append(theta[i])
        vec_theta = np.array(vec_theta)
        for sn in range(self.nsn):
            
            var[sn] = np.dot(np.dot(vec_theta,self.cov[sn]), vec_theta.reshape((len(vec_theta),1))) + self.sigma_int**2 + self.dmz[sn]**2 
#On a retire sigma_b = 0.03

        self.residuals = residuals
        self.var = var
        self.chi2 = np.sum(self.residuals**2 / var)

        return self.chi2

    def chi2_dof(self, sigma_int):
        self.sigma_int = sigma_int
        chi2 = self.comp_chi2(self.theta)
        return (chi2/self.dof) - 1             

    def minimize(self):

        p0 = np.zeros(self.ntheta)
        p0[-1] = -19.2

        self.theta = optimize.fmin(self.comp_chi2, p0)
        c = self.chi2_dof(self.sigma_int)
        count = 0
        print('before iteration', c+1)
        print(c)
       # while c>1E-2:
        #    print('during_iteration', c+1)

#            self.sigma_int = optimize.fsolve(self.chi2_dof, 0.1)

 #           self.theta = optimize.fmin(self.comp_chi2, p0)
  #          c = (self.comp_chi2(self.theta) / self.dof) - 1.
            
   #         count += 1
    #        if count > 10:
     #           break
            

        self.results = c, self.sigma_int, self.theta
        print(c)
        print(self.sigma_int)
        print(self.theta)


    def plot_results(self, param_name, model='SALT2', host_properties='$\log(M/M_{\odot})$'):

        self.minimize()
        mu = copy.deepcopy(self.mb)
        mu += - self.theta[-1]
        mu_ajuste = copy.deepcopy(mu)
            
                
        for k in range(len(self.data[0])):            
            mu_ajuste -= self.theta[k]*self.data[:,k]
            
            mb_reduit_k = copy.deepcopy(self.mb) - distance_modulus(self.zcmb)
            
            for i in range(len(self.data[0])):
                if i!=k :
                    mb_reduit_k -= self.theta[i]*self.data[:,i]
                
            plt.figure(k)
            plt.scatter(self.data[:,k], mb_reduit_k, color='r', marker ='o', s = 20,linewidth=1)
            
            fit_mb_k=self.theta[k]*self.data[:,k]+self.theta[-1]
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
                plt.scatter(self.host_prop[0], self.data[:,k], c=self.p_host[0], cmap = 'bwr', marker ='o', s = 20,linewidth=1) 
                plt.errorbar(self.host_prop[0], self.data[:,k], linestyle='',                            
			                yerr = np.sqrt(self.cov[:,k+1,k+1]), 
                            xerr = [self.host_prop_err_minus[0], self.host_prop_err_sup[0]],
			                ecolor='grey', 
			                alpha=0.4, marker='.', 
			                zorder=0)
                plt.title('Parameter %s as a function of host galaxy mass'%(param_name[k]))
                plt.grid()
                plt.xlabel('$\log_{10} \\frac{M_{galaxy}}{M_{\odot}}$', fontsize=12)
                plt.ylabel('Parameter %s'%(param_name[k]), fontsize = 12)
                cb = plt.colorbar()
                cb.set_label('Probability of having this mass')
                plt.show()

            
                plt.scatter(self.host_prop[1], self.data[:,k], c=self.p_host[1], cmap = 'bwr', marker ='o', s = 20,linewidth=1)
                plt.errorbar(self.host_prop[1], self.data[:,k], linestyle='',
			                yerr = np.sqrt(self.cov[:,k+1,k+1]),
                            xerr = [self.host_prop_err_minus[1], self.host_prop_err_sup[1]],
			                ecolor='grey', 
			                alpha=0.4, marker='.', 
			                zorder=0)
                plt.title('Parameter %s as a function of local specific star formation rate'%(param_name[k]))
                plt.grid()
                plt.xlabel('Local sSFR', fontsize = 12)
                plt.ylabel('Parameter %s'%(param_name[k]), fontsize = 12)
                cb = plt.colorbar()
                cb.set_label('Probability of having this local sSFR')
                plt.show()

        z_span = np.linspace(1E-2,0.15,100)
        mu_th = distance_modulus(z_span)


    
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
        weighted_std = weighted_avg_and_std(self.residuals, np.sqrt(self.var))
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
#Trace de residus vs Masse
            gs=gridspec.GridSpec(1,2, width_ratios=[1,1])
            
            mu_no_correction = self.residuals + self.theta[-2]*self.p_host[0] 


            plt.subplot(gs[0])
            plt.scatter(self.host_prop[0], self.residuals, c=self.p_host[0], cmap = 'bwr', marker ='o', s = 20,linewidth=1)
            plt.title('Hubble diagram residuals with corrected mass step')
            plt.errorbar(self.host_prop[0], self.residuals, linestyle='',    
			            yerr=np.sqrt(self.var),
                        xerr = [self.host_prop_err_minus[0], self.host_prop_err_sup[0]],
			            ecolor='grey', 
			            alpha=0.4, marker='.', 
			            zorder=0)
            plt.grid()
            plt.ylim([-0.6, 0.4])
            plt.xlabel('$\log_{10} \\frac{M_{galaxy}}{M_{\odot}}$', fontsize = 12)
            plt.ylabel('$\Delta \mu$', fontsize = 12)
            
            
            plt.subplot(gs[1])
            plt.scatter(self.host_prop[0], mu_no_correction, c=self.p_host[0], cmap = 'bwr', marker ='o', s = 20,linewidth=1)
            plt.title('Hubble diagram residuals without corrected mass step')
            plt.errorbar(self.host_prop[0], mu_no_correction, linestyle='',
			            yerr=np.sqrt(self.var),
                        xerr = [self.host_prop_err_minus[0], self.host_prop_err_sup[0]],
			            ecolor='grey', 
			            alpha=0.4, marker='.', 
			            zorder=0)
            plt.grid()
           # plt.ylim([-0.6, 0.4])
            #plt.yticks([-0.6, -0.4, -0.2, 0., 0.2, 0.4],[])
            plt.xlabel('$\log_{10} \\frac{M_{galaxy}}{M_{\odot}}$', fontsize = 12)
            cb = plt.colorbar()
            cb.set_label('Probability of having this mass')
            
            plt.subplots_adjust(wspace=0)
            plt.show()

#Trace de residus vs LSSFR
            plt.scatter(self.host_prop[1], self.residuals, c=self.p_host[1], cmap = 'bwr', marker ='o', s = 20,linewidth=1)
            plt.errorbar(self.host_prop[1], self.residuals, linestyle='',
			            yerr=np.sqrt(self.var),
                        xerr = [self.host_prop_err_minus[1], self.host_prop_err_sup[1]],
			            ecolor='grey', 
			            alpha=0.4, marker='.', 
			            zorder=0)
            plt.title('Hubble diagram residuals as a function of local specific star formation rate')
            plt.grid()
            plt.xlabel('Local sSFR', fontsize = 12)
            plt.ylabel('$\Delta \mu$', fontsize = 12)  
            cb = plt.colorbar()
            cb.set_label('Probability of having this local sSFR')
            plt.show()

        
    
if __name__ == "__main__":
    file_host = '../snfactory/lssfr_paper_full_sntable.csv'
    
    #mb, params, cov ,zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up= load_salt2('../snfactory/data_complete/sugar_companion_dataset.pkl', file_host)
    #hd = hubble_diagram(mb, params, cov ,zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up)
    #hd.plot_results(['X1', 'X', 'Probability of having this mass'])

    #mb, params, cov ,zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up= load_sugar_data()
    #hd = hubble_diagram(mb, params, cov ,zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up)
    #hd.plot_results(['q1', 'q2','q3', 'Av'])


    z, mb, data, data_cov, mass, proba = generate_mc_data(N=10000, Mb=-19.2, alphas=[-0.15, 3.8], stds=[1., 0.1],
                                                          mb_err=0.03, stds_err = [0.05, 0.01], step=0.12)
    print(mb)    
    dmz=np.zeros(len(mb))
    hd = hubble_diagram(mb, data, data_cov ,z, dmz)
    hd.plot_results(['q1', 'q2'])







    

