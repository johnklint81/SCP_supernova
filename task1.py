import numpy as np
import pandas as pd
import emcee
from scipy.stats import gamma, invgamma, t, norm, uniform, norminvgauss, mode
import corner
import matplotlib.pyplot as plt
import time

c = 1
n_parameters = 2
a0 = - n_parameters / 2
b0 = 0.1
n_walkers = 100
n_dim = 3
burn_in = 100
thinning = 1

datafile = open('SCPUnion2.1_mu_vs_z.txt', 'r')
SCP_data = pd.read_table(datafile, comment='#',
                         names=['SN name', 'Redshift', 'Distance_modulus',
                                'Distance_modulus_error', 'P_low_mass'])

SCP_data_small_z = SCP_data[SCP_data.Redshift < 0.5]
small_z = SCP_data_small_z.Redshift
small_z = small_z.to_numpy()

small_z_distance_modulus = SCP_data_small_z.Distance_modulus
small_z_distance = 10 ** ((small_z_distance_modulus - 25) / 5)  # dL
small_z_distance = small_z_distance.to_numpy()

small_z_errors = SCP_data_small_z.Distance_modulus_error
small_z_errors = small_z_errors.to_numpy()

W = 1 / small_z_errors
W /= np.sum(W)
variance_small_z_data = np.var(small_z_distance)

small_z_distance *= W

# [H0, q0, variance]
start_position = [65, 0, variance_small_z_data] + [5, 1, np.sqrt(variance_small_z_data)] \
                 * np.random.randn(n_walkers, n_dim)


def log_likelihood(_z, _dL, _parameters, _c):
    _H0, _q0, _variance = _parameters
    _model = _c / _H0 * (_z + 0.5 * (1 - _q0) * _z ** 2)
    _n_datapoints = len(_dL)
    _log_likelihood = - 0.5 * (np.sum((_dL - _model) ** 2 / _variance)) - (_n_datapoints / 2) * np.log(_variance)
    return _log_likelihood


def log_prior(_parameters, _a0, _b0):
    _H0, _q0, _variance = _parameters
    _log_inverseGamma = invgamma.logpdf(_variance, a=_a0, scale=_b0)
    _log_prior = _log_inverseGamma
    return _log_prior


def log_posterior(_parameters, _z, _dL, _a0, _b0, _c):

    _log_prior = log_prior(_parameters, _a0, _b0)
    if not np.isfinite(_log_prior):
        return -np.inf
    _log_likelihood = log_likelihood(_parameters, _z, _dL, _c)
    _log_posterior = _log_prior + _log_likelihood
    return _log_posterior


start = time.time()
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_posterior, args=(small_z, small_z_distance, a0, b0, c))
sampler.run_mcmc(start_position, 2000, progress=True)
end = time.time()
serial_time = end - start
print("Serial took {0:.1f} seconds".format(serial_time))
flat_mcmc_samples = sampler.get_chain(discard=burn_in, thin=thinning, flat=True)
fig = corner.corner(flat_mcmc_samples, labels=["$H_0$", "$q_0$", "$\sigma^2$"], show_titles=True)

plt.show()
