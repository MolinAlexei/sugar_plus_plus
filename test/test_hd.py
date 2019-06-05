import numpy as np
import sugar_plus_plus as spp
from spp_test_helper import timer

def generate_mc_data(N=10000, Mb=-19.2, alphas=[-0.15, 3.8], stds=[1., 0.1],
                     mb_err=0.03, stds_err = [0.05, 0.01], step=0.12,
                     disp_int=0, dz = 0.001):

    np.random.seed(13)
    z = np.random.uniform(0.01, 0.1, size = N)
    mu = spp.distance_modulus(z)
    z += np.random.normal(scale=dz, size=N)
    dmz = (5. / np.log(10)) * dz / z
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

    if disp_int > 0:
        mb += np.random.normal(scale=disp_int, size=N)


    mass = np.random.uniform(5,15, size=N)
    filter_higth_mass = (mass > 10)
    proba = np.zeros(N)
    proba[filter_higth_mass] = 1.
    mb += proba * step
    
    truth_theta = [alpha for alpha in alphas]
    truth_theta.append(step)
    truth_theta.append(Mb)

    return z, dmz, mb, data, data_cov, mass, proba, np.array(truth_theta), disp_int

@timer
def test_hubble_diagram():

    z, dmz, mb, data, data_cov, mass, proba, truth_theta, truth_disp_int = generate_mc_data(N=3000, Mb=-19.2,
                                                                                            alphas=[-0.15, 3.8],
                                                                                            stds=[1, 0.1],
                                                                                            mb_err=0.03,
                                                                                            stds_err = [0.05, 0.01],
                                                                                            step=0.08,
                                                                                            disp_int=0.1,
                                                                                            dz = 0.001)
    hd = spp.hubble_diagram(mb, data, data_cov ,z, dmz=dmz, host_prop=mass, p_host=proba)
    hd.minimize()

    np.testing.assert_allclose(hd.theta, truth_theta, atol=5e-2)
    np.testing.assert_allclose(hd.sigma_int, truth_disp_int, atol=5e-2)

if __name__ == "__main__":

    test_hubble_diagram()
