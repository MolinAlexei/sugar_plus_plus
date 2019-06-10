""" Math toolbox, that include all the usefull tools."""

import numpy as np

def comp_rms(residuals, dof, err=True, variance=None):
    """
    Compute the RMS or WRMS of a given distribution.

    :param 1D-array residuals: the residuals of the fit.
    :param int dof: the number of degree of freedom of the fit.
    :param bool err: return the error on the RMS (WRMS) if set to True.
    :param 1D-aray variance: variance of each point. If given,
                             return the weighted RMS (WRMS).

    :return: rms or rms, rms_err
    """
    if variance is None:                # RMS
        rms = float(np.sqrt(np.sum(residuals**2)/dof))
        rms_err = float(rms / np.sqrt(2*dof))
    else:                               # Weighted RMS
        assert len(residuals) == len(variance)
        rms = float(np.sqrt(np.sum((residuals**2)/variance) / np.sum(1./variance)))
        #rms_err = float(N.sqrt(1./N.sum(1./variance)))
        rms_err = np.sqrt(2.*len(residuals)) / (2*np.sum(1./variance)*rms)
    if err:
        return rms, rms_err
    else:
        return rms

def flbda2fnu( x, y, var=None, backward=False):
    """Convert *x* [A], *y* [erg/s/cm2/A] to *y* [erg/s/cm2/Hz]. Se
    `var=var(y)` to get variance."""

    f = x**2 / 299792458. * 1.e-10 # Conversion factor                                                                                                                      
    if backward: 
        f = 1./f   
    if var is None:                # Return converted signa
        return y * f
    else:                          # Return converted variance
        return var * f**2

def flbda2ABmag( x, y, ABmag0=48.59, var=None):
    """Convert *x* [A], *y* [erg/s/cm2/A] to `ABmag =                                                                                                                       
    -2.5*log10(erg/s/cm2/Hz) - ABmag0`. Set `var=var(y)` to get                                                                                                             
    variance."""          
    z = flbda2fnu(x,y)
    if var is None:
        return -2.5*np.log10(z) - ABmag0 
    else:
        return (2.5/np.log(10)/z)**2 * flbda2fnu(x,y,var=var)
    