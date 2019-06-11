from matplotlib import pyplot as plt
import copy
import numpy as np
from scipy.linalg import block_diag
from numpy.linalg import inv
from scipy import optimize
try:
   import cPickle as pkl
except ModuleNotFoundError:
   import pickle as pkl
import iminuit as minuit
import sugar_plus_plus as spp
from sugar_plus_plus import distance_modulus_th
from sugar_plus_plus import comp_rms


def make_method(obj):
    """Decorator to make the function a method of *obj*.
    In the current context::
      @make_method(Axes)
      def toto(ax, ...):
          ...
    makes *toto* a method of `Axes`, so that one can directly use::
      ax.toto()
    COPYRIGHT: from Yannick Copin
    """

    def decorate(f):
        setattr(obj, f.__name__, f)
        return f

    return decorate


def get_hubblefit(x, cov_x, zhl, zcmb, dmz, PARAM_NAME=np.asarray(['alpha1','alpha2',"alpha3","beta","delta", "delta2", "delta3"]), lssfr=None, sn_name=None):
    """
    Parameters
    ----------
    x: type
        infor
    
    cov_x ....
    
    
    parameters: list of int to specify if you want to remove some parameters in the fit in PARAM_NAME. 
                By default is None and this take all parameters. If you don't want to put a correction use 
                parameters=[]
        example: 
            for usual salt x1 c correction if you want only color correction, use parameters = [2]. 
            Warning if you have more than 5 parameters, add parameters in PARAM_NAME
    """

    n_corr =  np.shape(x)[1]-1 
    class hubble_fit_case(Hubble_fit):
        freeparameters = ["Mb"]+PARAM_NAME[:n_corr].tolist()
    h = hubble_fit_case(x, cov_x, zhl, zcmb, dmz, lssfr=lssfr, sn_name=sn_name)
    return h


class Hubble_fit(object):
    """
    """
    
    def __new__(cls,*arg,**kwargs):
        """ Upgrade of the New function to enable the
        the _minuit_ black magic
        """
        obj = super(Hubble_fit,cls).__new__(cls)
        
        exec ("@make_method(Hubble_fit)\n"+\
             "def _minuit_chi2_(self,%s): \n"%(", ".join(obj.freeparameters))+\
             "    parameters = %s \n"%(", ".join(obj.freeparameters))+\
             "    return self.get_chi2(parameters)\n")


        return obj

    def __init__(self, X, cov_X, zhl, zcmb, dmz,  guess=None, lssfr=None, sn_name=None):
        self.variable = X
        self.cov = cov_X
        self.zcmb = zcmb
        self.zhl = zhl
        #self.zerr = zerr
        self.dmz = dmz #5/np.log(10) * np.sqrt(self.zerr**2 + 0.001**2) / self.zcmb #adding peculiar velocity
        self.sn_name = sn_name
        self.dof = len(X)-len(self.freeparameters)  

        if lssfr is not None:
            self.lssfr = lssfr[:,0]
            self.proba = lssfr[:,1]
        else:
            self.lssfr = None
            self.proba = None
    

    def distance_modulus(self, params):
        """
        (mb + alpha * v1 + beta * v2 .....) - Mb
        """

        return  np.sum(np.concatenate([[1],params[1:]]).T * self.variable, axis=1) - params[0]     
    
    def get_chi2(self, params):
        """
        """
        self.Cmu = np.zeros_like(self.cov[::len(params),::len(params)])
        pcorr = np.concatenate([[1],params[1:]])

        for i, coef1 in enumerate(pcorr):
            for j, coef2 in enumerate(pcorr):
                self.Cmu += (coef1 * coef2) * self.cov[i::len(params), j::len(params)] 
                
                
        self.Cmu[np.diag_indices_from(self.Cmu)] += self.sig_int**2 + self.dmz**2 
        self.C = inv(self.Cmu)
        self.distance_modulus_table =  self.distance_modulus(params)
        L = self.distance_modulus_table - distance_modulus_th(self.zcmb, self.zhl)
        self.residuals = L
        self.var = np.diag(self.Cmu)
        the_chi2 = np.dot(L, np.dot(self.C,L))
        print the_chi2
        return the_chi2 


    def fit_intrinsic(self, intrinsic_guess=0.1):
        """ Get the most optimal intrinsic dispersion given the current fitted standardization parameters. 
        
        The optimal intrinsic magnitude dispersion is the value that has to be added in quadrature to 
        the magnitude errors such that the chi2/dof is 1.
        Returns
        -------
        float (intrinsic magnitude dispersion)
        """
        def get_intrinsic_chi2dof(intrinsic):
            self.sig_int = intrinsic
            return np.abs(self.get_chi2(self.resultsfit) / self.dof -1)
        
        return optimize.fmin(get_intrinsic_chi2dof,
                             intrinsic_guess)

    def setup_guesses(self,**kwargs):
        """ Defines the guesses, boundaries and fixed values
        that will be passed to the given model.
        For each variable `v` of the model (see freeparameters)
        the following array will be defined and set to param_input:
           * v_guess,
           * v_boundaries,
           * v_fixed.
        Three arrays (self.paramguess, self.parambounds,self.paramfixed)
        will be accessible that will point to the defined array.
        Parameter
        ---------
        **kwargs the v_guess, v_boundaries and, v_fixed for as many
        `v` (from the freeparameter list).
        All the non-given `v` values will be filled either by pre-existing
        values in the model or with: 0 for _guess, False for _fixed, and
        [None,None] for _boundaries
        Return
        ------
        Void, defines param_input (and consquently paramguess, parambounds and paramfixed)
        """
        def _test_it_(k,info):
            param = k.split(info)[0]
            if param not in self.freeparameters:
                raise ValueError("Unknown parameter %s"%param)

        self.param_input = {}
        # -- what you hard coded
        for name in self.freeparameters:
            for info in ["_guess","_fixed","_boundaries"]:
                if hasattr(self, name+info):
                    self.param_input[name+info] = eval("self.%s"%(name+info))
                    
        # -- Then, whatever you gave
        for k,v in kwargs.items():
            if "_guess" in k:
                _test_it_(k,"_guess")
            elif "_fixed" in k:
                _test_it_(k,"_fixed")
            elif "_boundaries" in k:
                _test_it_(k,"_boundaries")
            else:
                raise ValueError("I am not able to parse %s ; not _guess, _fixed nor _boundaries"%k)
            self.param_input[k] = v

        # -- Finally if no values have been set, let's do it
        for name in self.freeparameters:
            if name+"_guess" not in self.param_input.keys():
                self.param_input[name+"_guess"] = 0.
            if name+"_fixed" not in self.param_input.keys():
                self.param_input[name+"_fixed"] = False
            if name+"_boundaries" not in self.param_input.keys():
                self.param_input[name+"_boundaries"] = [None,None]
    
    def fit(self, fit_intrinsic=True, **kwargs):
        """
        How to use kwargs 
        For each variable `v` of the model (see freeparameters)
        the following array will be defined and set to param_input:
           * v_guess,
           * v_boundaries,
           * v_fixed.
        Three arrays (self.paramguess, self.parambounds,self.paramfixed)
        will be accessible that will point to the defined array.
        Parameter
        ---------
        **kwargs the v_guess, v_boundaries and, v_fixed for as many
        `v` (from the freeparameter list).
        All the non-given `v` values will be filled either by pre-existing
        values in the model or with: 0 for _guess, False for _fixed, and
        [None,None] for _boundaries
        """
        self._loopcount = 0
        self.sig_int = 0.
        self.setup_guesses(**kwargs)
        
        self.first_iter = self._fit_minuit_()
        # - Intrinsic disposerion Fit?
        if fit_intrinsic:
            while (np.abs(self.chi2_per_dof - 1) > 0.001 and self._loopcount < 30):
                self.sig_int =  self.fit_intrinsic(np.sqrt(np.mean(self.var))*2. / self.chi2_per_dof)
                self._fit_minuit_()
                self._loopcount += 1
                
        # - Final steps      
        return self._fit_readout_()

        
    def _setup_minuit_(self):
        """
        """
        # == Minuit Keys == #
        minuit_kwargs = {}
        for param in self.freeparameters:
            minuit_kwargs[param] = self.param_input["%s_guess"%param]
            

        self.minuit = minuit.Minuit(self._minuit_chi2_, **minuit_kwargs)
    
    def _fit_minuit_(self):
        """
        """
        self._setup_minuit_()
        self._migrad_output_ = self.minuit.migrad()
        
        if self._migrad_output_[0]["is_valid"] is False:
            print("migrad is not valid")
            
            
        self.resultsfit = np.asarray([self.minuit.values[k]
                              for k in self.freeparameters])
        self.chi2_per_dof = self.minuit.fval/self.dof
        
    def _fit_readout_(self):
        """ Computes the main numbers """
        return comp_rms(self.residuals, self.dof, err=True, variance=self.var) , self.sig_int  
    
    def comp_mass_step_modefit(self, xcut=-10.8, xlabel='$\log(lsSFR)$', PRINT_WRMS=False, model_name=None, fig_path='../../lsfr_analysis/lsfr_') :
        import modefit
        if (self.proba == self.variable[:,-1]).all():
            raise ValueError('Step done in the gloabal fit')
        if model_name == None:
            raise ValueError('model_name have to be salt2 or sugar')
        self.fit(Mb_guess = -19.05)
        print ('chi2', self.minuit.fval, 'dof ', self.dof)
        print ('chi2 per dof ', self.chi2_per_dof )
        step = modefit.stepfit(self.lssfr, self.residuals, np.sqrt(self.var-0.1**2),
                               proba=self.proba, dx=None, xcut=xcut, masknan=True, names=None)
        step.fit()
        FIG = plt.figure(figsize=(9,6))
        PLOT = step.show(figure= FIG)
        ax = PLOT['ax'][0]
        ax.set_ylim(-0.6,0.6)
 

        ylabel='$\Delta \mu$ '+ model_name
        ax.set_ylabel(ylabel,fontsize=16)
        ax.set_xlabel(xlabel,fontsize=16)
        PLOT['fig'].subplots_adjust(top=0.97,right=0.99)
        print ('sig int ', self.sig_int)
        import sugar_training as st
        wrms, wrms_err = st.comp_rms(self.residuals, 1, variance=self.var)

        if PRINT_WRMS:
            ax.set_title('$wRMS = (%.2f \pm %.2f)$ mag, $\Delta_Y=(%.2f \pm %.2f)$ mag'%((wrms, wrms_err,
                                                                                          abs(step.modelstep[0]), 
                                                                                          step.modelstep[1])),
                         fontsize=14)
        else:
            ax.set_title('$\Delta_Y=(%.2f \pm %.2f)$ mag'%((abs(step.modelstep[0]), 
                                                            step.modelstep[1])),
                         fontsize=18)

        print (model_name+' step: ', step.modelstep)
        plt.savefig(fig_path+model_name+'.png')
        self.step_fitvalues = step.fitvalues


       
    def dump_pkl(self, outpath='../../lsfr_analysis/HD_results_sugar.pkl'):
        HD_results = {}
        err = np.sqrt(np.diag(self.cov))
        HD_results['minuit_results'] = self.minuit.values
        HD_results['minuit_results_errors'] = self.minuit.errors
        HD_results['modefit_values'] =  self.step_fitvalues
        HD_results['sig_int'] = self.sig_int
        HD_results['data'] = {}
        for i, name in enumerate (self.sn_name):
            HD_results['data'][name] = {}
            HD_results['data'][name]['residuals'] = self.residuals[i]
            if self.lssfr is not None:
                HD_results['data'][name]['lssfr'] = self.lssfr[i]
                HD_results['data'][name]['proba'] = self.proba[i]
            HD_results['data'][name]['var'] = self.var[i]
            
            HD_results['data'][name]['mb'] = self.variable[i,0]
           
            HD_results['data'][name]['mb_err'] = err[i*len(self.variable[0,:])]
            
            for l in range(len(self.variable[0,:-1])):
                HD_results['data'][name]['param_'+str(l+1)] = self.variable[i,l+1]
                HD_results['data'][name]['param_'+str(l+1)+'_err'] = []
                HD_results['data'][name]['param_'+str(l+1)+'_err'] = err[i*len(self.variable[0,:])+l+1]
               
            
                
        pkl.dump(HD_results, open(outpath, 'w'))

if __name__ == "__main__":

   file_host = '../../../../lssfr_paper_full_sntable.csv'
   file_salt2 = '../../../../../sands_companion_dataset/sugar_companion_dataset.pkl'
   file_sugar = '../../../../../sugar_analysis/sugar_analysis/meta_sugar.yaml'

   #### SALT2 fit ####

   mb, params, cov, zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up = spp.load_salt2(file_salt2, file_host)
   
   data = np.zeros((len(params[:,0]), len(params[0])+2))
   data[:,0] = mb
   for i in range(len(params[0])):
      data[:,i+1] = params[:,i]
   data[:,-1] = 1-p_host[1]
   
   COV = np.zeros((4*len(mb), 4*len(mb)))
   for sn in range(len(mb)):
      COV[sn*4:((sn+1)*4)-1, sn*4:((sn+1)*4)-1] = cov[sn]
   
   hf_salt2 = get_hubblefit(data, COV, zcmb, zcmb, dmz, 
                            PARAM_NAME=np.asarray(['alpha', 'beta', 'step_lsSFR']), 
                            lssfr=np.array([host_prop[1], p_host[1]]).T, sn_name=None)
   hf_salt2.fit(Mb_guess = -19.2)

   #### SUGAR fit ###

   mb, params, cov, zcmb, dmz, host_prop, p_host, host_prop_err_down, host_prop_err_up = spp.load_sugar_data(file_sugar,
                                                                                                             file_host)
   
   data = np.zeros((len(params[:,0]), len(params[0])+2))
   data[:,0] = mb
   for i in range(len(params[0])):
      data[:,i+1] = params[:,i]
   data[:,-1] = 1-p_host[1]
   
   COV = np.zeros((6*len(mb), 6*len(mb)))
   for sn in range(len(mb)):
      COV[sn*6:((sn+1)*6)-1, sn*6:((sn+1)*6)-1] = cov[sn]
   
   hf_sugar = get_hubblefit(data, COV, zcmb, zcmb, dmz, 
                      PARAM_NAME=np.asarray(['a1', 'a2', 'a3', 'b', 'step_lsSFR']), 
                      lssfr=np.array([host_prop[1], p_host[1]]).T, sn_name=None)
   hf_sugar.fit(Mb_guess = -19.2)
