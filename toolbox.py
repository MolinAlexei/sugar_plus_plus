import numpy as np
import pylab as plt


def luminosity_distance(z):
    c = 299792.458
    H0 = 70.
    Om = 0.3
    c_H0 = (c / H0) * 1e6
    formule = (z + (1. - (0.75 * Om)) * z**2)
    return c_H0 * formule

def distance_modulus(z):
    return 5. * np.log10(luminosity_distance(z)) - 5.

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise
    return np.sqrt(variance)

if __name__ == '__main__':

    redshift = np.linspace(0.01, 0.08, 100)
    plt.plot(redshift, distance_modulus(redshift))
