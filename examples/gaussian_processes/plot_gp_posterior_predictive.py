# -*- coding: utf-8 -*-
"""
Gaussian Process Posterior Predictive
=====================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 3

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from etudes.datasets import synthetic_sinusoidal, make_regression_dataset
from etudes.gaussian_processes import (gp_sample_custom,
                                       dataframe_from_gp_samples,
                                       dataframe_from_gp_summary)

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# constants
num_train = 25  # nbr training points in synthetic dataset
num_features = 1  # dimensionality
num_index_points = 256  # nbr of index points
num_samples = 7

observation_noise_variance = 1e-1
jitter = 1e-6

kernel_cls = kernels.ExponentiatedQuadratic

seed = 42  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

golden_ratio = 0.5 * (1 + np.sqrt(5))

X_pred = np.linspace(-1.0, 1.0, num_index_points).reshape(-1, num_features)

load_data = make_regression_dataset(synthetic_sinusoidal)
X_train, Y_train = load_data(num_train, num_features,
                             observation_noise_variance,
                             x_min=-0.5, x_max=0.5,
                             random_state=random_state)

# %%
# Synthetic dataset
# -----------------

fig, ax = plt.subplots()

ax.plot(X_pred, synthetic_sinusoidal(X_pred), label="true")
ax.scatter(X_train, Y_train, marker='x', color='k',
           label="noisy observations")

ax.legend()

ax.set_xlim(-0.6, 0.6)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.show()

# %%
# Schur Complement kernel matrix
# ------------------------------

base_kernel = kernel_cls()
kernel = kernels.SchurComplement(
    base_kernel, X_train,
    diag_shift=observation_noise_variance
)

fig, ax = plt.subplots()


ax.imshow(kernel.matrix(X_pred, X_pred), extent=[-1.0, 1.0, 1.0, -1.0],
          cmap="cividis")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$x$')

plt.show()

# %%
# Posterior predictive distribution
# ---------------------------------

gprm = tfd.GaussianProcessRegressionModel(
    kernel=kernel, index_points=X_pred,
    observation_index_points=X_train,
    observations=Y_train,
    observation_noise_variance=observation_noise_variance,
    jitter=jitter
)

# %%


def fill_between_stddev(X_pred, mean_pred, stddev_pred, n=1, ax=None, **kwargs):

    if ax is None:
        ax = plt.gca()

    ax.fill_between(X_pred,
                    mean_pred - n * stddev_pred,
                    mean_pred + n * stddev_pred, **kwargs)

# %%


fig, ax = plt.subplots()

ax.plot(X_pred, synthetic_sinusoidal(X_pred), label="true")
ax.scatter(X_train, Y_train, marker='x', color='k',
           label="noisy observations")

ax.plot(X_pred, gprm.mean(), color="tab:orange",
        label="posterior predictive mean")

fill_between_stddev(X_pred.squeeze(),
                    gprm.mean().numpy().squeeze(),
                    gprm.stddev().numpy().squeeze(),
                    alpha=0.1, color="tab:orange",
                    label="posterior predictive std dev", ax=ax)

ax.legend()

ax.set_xlim(-0.6, 0.6)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')

plt.show()
