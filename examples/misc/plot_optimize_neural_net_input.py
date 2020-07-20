# -*- coding: utf-8 -*-
"""
Optimize Neural Network Input
=============================

We study how to optimize the output of a neural network with respect to the
input.

"""
# sphinx_gallery_thumbnail_number = 4
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras.backend as K

import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import GlorotUniform
# from tensorflow.keras.regularizers import l2

from scipy.optimize import minimize

# from etudes.losses import binary_crossentropy_from_logits
from etudes.datasets import make_regression_dataset
from etudes.decorators import unbatch, value_and_gradient, numpy_io
# %%

K.set_floatx("float64")

num_samples = 20
num_features = 1

num_index_points = 512
xmin, xmax = -1.0, 2.0
X_grid = np.linspace(xmin, xmax, num_index_points).reshape(-1, num_features)

noise_variance = 0.1
learning_rate = 0.01

sparsity_factor = 0.1
step = int(1/sparsity_factor)
X_grid_sparse = X_grid[::step]

seed = 8989
random_state = np.random.RandomState(seed)
# %%


def latent(x):
    # return (6.0*x-2.0)**2 * np.sin(12.0*x - 4.0)
    return np.sin(3.0*x) + x**2 - 0.7*x
# %%


load_observations = make_regression_dataset(latent)
X, y = load_observations(num_samples=num_samples,
                         num_features=num_features,
                         noise_variance=noise_variance,
                         x_min=xmin, x_max=xmax,
                         random_state=random_state)
# %%
fig, ax = plt.subplots()

ax.plot(X_grid, latent(X_grid), label="true", color="tab:gray")
ax.scatter(X, y, marker='x', color='k', label="noisy observations")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r"$y$")

ax.legend()

plt.show()
# %%
model = Sequential([
    Dense(32, activation="relu", kernel_initializer=GlorotUniform(seed=seed)),
    Dense(32, activation="relu", kernel_initializer=GlorotUniform(seed=seed)),
    Dense(1, kernel_initializer=GlorotUniform(seed=seed))
])
model.compile(optimizer="adam",
              loss="mean_squared_error",
              metrics=["mean_squared_error"])
model.fit(X, y, epochs=2000, batch_size=100)
# %%
fig, ax = plt.subplots()

ax.plot(X_grid, latent(X_grid), label="true", color="tab:gray")
ax.plot(X_grid, model(X_grid), label="predicted")

ax.scatter(X, y, marker='x', color='k', label="noisy observations")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r"$y$")

ax.legend()

plt.show()

# %%
val, grad = tfp.math.value_and_gradient(model, X_grid_sparse)
new_val = model(X_grid_sparse - learning_rate * grad)
# %%

fig, ax = plt.subplots()

ax.plot(X_grid, model(X_grid), label="value")
ax.quiver(X_grid_sparse, val, - learning_rate * grad, new_val - val,
          scale_units='xy', angles='xy', scale=1.0, label="gradient")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

ax.legend()

plt.show()
# %%


@value_and_gradient
@unbatch
def func(x):

    return model(x)
# %%


initial_position = random_state.uniform(low=xmin, high=xmax, size=(1,))
# %%
numpy_io(func)(initial_position)
# %%
result = tfp.optimizer.lbfgs_minimize(func, initial_position=initial_position)
result
# %%
result.converged
# %%
result.failed
# %%
result.num_iterations
# %%
result.num_objective_evaluations
# %%
result.position
# %%
result.objective_value
# %%
result.objective_gradient
# %%
fig, ax = plt.subplots()


ax.scatter(initial_position, model(initial_position.reshape(1, -1)), s=8**2,
           c="tab:orange", marker='o', alpha=0.8, zorder=10, label="initial")
ax.scatter(result.position, tf.expand_dims(result.objective_value, axis=0), s=8**2,
           c="tab:orange", marker='*', alpha=0.8, zorder=10, label="minimum")
ax.plot(X_grid, model(X_grid), label="value")
ax.quiver(X_grid_sparse, val, - learning_rate * grad, new_val - val,
          scale_units='xy', angles='xy', scale=1.0, label="gradient")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

ax.legend()

plt.show()
# %%
# SciPy Optimize

x_hist = [initial_position]
minimize(numpy_io(func), x0=initial_position, jac=True,
         method="L-BFGS-B", tol=1e-8, callback=x_hist.append)
# %%
X_hist = np.vstack(x_hist)
Y_hist = model(X_hist)
# %%
fig, ax = plt.subplots()

ax.plot(X_grid, model(X_grid), label="value")
ax.quiver(X_hist[:-1], Y_hist[:-1],
          X_hist[1:] - X_hist[:-1],
          Y_hist[1:] - Y_hist[:-1],
          scale_units='xy', angles='xy', scale=1.0, width=3e-3)

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$y$')

ax.legend()

plt.show()
