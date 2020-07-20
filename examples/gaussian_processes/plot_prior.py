# -*- coding: utf-8 -*-
"""
Gaussian Process Prior
======================

Hello world
"""
# sphinx_gallery_thumbnail_number = 3

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt

# %%

# shortcuts
tfd = tfp.distributions
kernels = tfp.math.psd_kernels

# constants
num_features = 1  # dimensionality
num_index_points = 256  # nbr of index points
num_samples = 8

x_min, x_max = -5.0, 5.0
X_grid = np.linspace(x_min, x_max, num_index_points).reshape(-1, num_features)

seed = 23  # set random seed for reproducibility
random_state = np.random.RandomState(seed)

kernel = kernels.ExponentiatedQuadratic()

# %%
# Kernel profile
# --------------
# The exponentiated quadratic kernel is *stationary*.
# That is, :math:`k(x, x') = k(x, 0)` for all :math:`x, x'`.

fig, ax = plt.subplots()

ax.plot(X_grid, kernel.apply(X_grid, np.zeros((1, num_features))))

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$k(x, 0)$')

plt.show()

# %%
# Kernel matrix
# -------------
fig, ax = plt.subplots()

ax.pcolormesh(*np.broadcast_arrays(X_grid, X_grid.T),
              kernel.matrix(X_grid, X_grid), cmap="cividis")
ax.invert_yaxis()

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$x$')

plt.show()

# %%
# Prior samples
# -------------

gp = tfd.GaussianProcess(kernel=kernel, index_points=X_grid)
samples = gp.sample(num_samples, seed=seed)
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, samples.numpy().T)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$f(x)$')
ax.set_title(r'Draws of $f(x)$ from GP prior')

plt.show()
