import numpy as np
import os
import pickle
import yaml
import sugar_plus_plus as spp

def load_salt2(File, File_host):

    table_host = np.loadtxt(File_host, comments='#', delimiter=',', dtype='str',skiprows=1)
    sn_name = table_host[:,16]
    
    #['HR', 'HR.err', 'HR_o', 'HR_o.err', 'gmass', 'gmass.err_down',
    #    'gmass.err_up', 'host.zcmb', 'host.zhelio', 'host.zhelio.err',
    #    'lssfr', 'lssfr.err_down', 'lssfr.err_up', 'mass', 'mass.err_down',
    #    'mass.err_up', 'name', 'p(highgmass)', 'p(prompt)', 'salt2.Color',
    #    'salt2.Color.err', 'salt2.CovColorRestFrameMag_0_B',
    #    'salt2.CovColorX1', 'salt2.CovRestFrameMag_0_BX1',
    #    'salt2.RestFrameMag_0_B', 'salt2.RestFrameMag_0_B.err', 'salt2.X1',
    #    'salt2.X1.err', 'target.dec', 'target.ra']
    
    X1 = table_host[:,-4].astype(float)
    X1_err = table_host[:,-3].astype(float)
    C = table_host[:,-11].astype(float)
    C_err = table_host[:,-10].astype(float)
    mb = table_host[:,2].astype(float) #+ spp.distance_modulus(table_host[:,7].astype(float))
    delta_mu_err = table_host[:,3].astype(float)

    delta_mu_X1_cov = table_host[:,-7].astype(float)
    delta_mu_C_cov = table_host[:,-9].astype(float)
    X1_C_cov = table_host[:,-8].astype(float)
    dmz = (5. / np.log(10)) * np.sqrt(table_host[:,9].astype(float)**2 + 0.001**2) / table_host[:,7].astype(float)
    zcmb = table_host[:,7].astype(float)
    z_err  =  table_host[:,9].astype(float)
    cov = np.zeros((len(X1), 3, 3))

    for k in range(len(X1)):
        cov[k] = np.array([[delta_mu_err[k]**2, delta_mu_X1_cov[k], delta_mu_C_cov[k]],
                           [delta_mu_X1_cov[k], X1_err[k]**2,  X1_C_cov[k]],
                           [delta_mu_C_cov[k], X1_C_cov[k], C_err[k]**2]])
    
    global_mass = table_host[:,4].astype(float)
    gmass_err_down = table_host[:,5].astype(float)
    gmass_err_up = table_host[:,6].astype(float)
    lssfr = table_host[:,10].astype(float)
    lssfr_err_down = table_host[:,11].astype(float)
    lssfr_err_up = table_host[:,12].astype(float)
    p_hightmass = table_host[:,17].astype(float)
    p_prompt = table_host[:,18].astype(float)

    params = np.array([X1, C]).T

    host_prop = [global_mass, lssfr]
    p_host = [p_hightmass, p_prompt]
    host_prop_err_down = [gmass_err_down, lssfr_err_down]
    host_prop_err_up = [gmass_err_up, lssfr_err_up]

    return mb, params, cov ,zcmb, z_err, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up


def load_salt2_old(File, File_host):

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


    for k, sn in enumerate(sn_name):
        for j, snh  in enumerate(sn_name_host):
            if sn == snh :
                global_mass[k] = table_host[j,4].astype(float)
                gmass_err_down[k] = table_host[j,5].astype(float)
                gmass_err_up[k] = table_host[j,6].astype(float)
                lssfr[k] = table_host[j,10].astype(float)
                lssfr_err_down[k] = table_host[j,11].astype(float)
                lssfr_err_up[k] = table_host[j,12].astype(float)
                p_hightmass[k] = table_host[j,17].astype(float)
                p_prompt[k] = table_host[j,18].astype(float)

    params = np.array([X1, C]).T

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

    dic = yaml.load(open(sn_data), Loader = yaml.Loader)
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

if __name__ == "__main__":

    file_host = '../../Data/lssfr_paper_full_sntable.csv'
    file_salt2 = '../../Data/sugar_companion_dataset.pkl'
    file_sugar = '../../Data/meta_sugar.yaml'

    KEY = 0

    mb, params, cov ,zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up= load_sugar_data(file_sugar, file_host)
    print(mb)
