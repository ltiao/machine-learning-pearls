# -*- coding: utf-8 -*-
"""
Divergence estimation with Gauss-Hermite Quadrature
===================================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 1

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

tfd = tfp.distributions

max_size = 300
num_seeds = 10

x_min, x_max = -5.0, 5.0

num_query_points = 256
num_features = 1

# query index points
X_pred = np.linspace(x_min, x_max, num_query_points)

# %%
# Example

p = tfd.Normal(loc=1.0, scale=1.0)
q = tfd.Normal(loc=0.0, scale=2.0)

# %%

fig, ax = plt.subplots()

ax.plot(X_pred, p.prob(X_pred), label='$p(x)$')
ax.plot(X_pred, q.prob(X_pred), label='$q(x)$')

ax.set_xlabel('$x$')
ax.set_ylabel('density')

ax.legend()

plt.show()

# %%
# Exact KL divergence (analytical)
# --------------------------------

kl_exact = tfd.kl_divergence(p, q).numpy()
kl_exact


# %%
# Approximate KL divergence (Monte Carlo)
# ---------------------------------------

sample_size = 25
seed = 8888

# %%

kl_monte_carlo = tfp.vi.monte_carlo_variational_loss(
    q.log_prob, p, sample_size=sample_size,
    discrepancy_fn=tfp.vi.kl_reverse, seed=seed).numpy()
kl_monte_carlo

# %%

X_samples = p.sample(sample_size, seed=seed)

# %%


def log_ratio(x):
    return q.log_prob(x) - p.log_prob(x)


def h(x):
    return tfp.vi.kl_reverse(log_ratio(x))

# %%


fig, ax = plt.subplots()

ax.plot(X_pred, h(X_pred), label='$f(x)$')
ax.scatter(X_samples, h(X_samples))

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$h(x)$')

plt.show()

# %%


def divergence_monte_carlo(q_log_prob_fn, p, sample_size,
                           discrepancy_fn=tfp.vi.kl_reverse, seed=None):
    # equivalent to:
    # tfp.vi.monte_carlo_variational_loss(
    #   q_log_prob_fn, p, sample_size,
    #   discrepancy_fn, seed
    # )

    def log_ratio(x):
        return q_log_prob_fn(x) - p.log_prob(x)

    def h(x):
        return discrepancy_fn(log_ratio(x))

    X_samples = p.sample(sample_size, seed=seed)

    # same as:
    # return tfp.monte_carlo.expectation(f=f, samples=X_samples)
    return tf.reduce_mean(h(X_samples), axis=-1)

# %%


divergence_monte_carlo(q.log_prob, p, sample_size, seed=seed).numpy()

# %%
# Approximate KL divergence (Gauss-Hermite Quadrature)
# ----------------------------------------------------
# Consider a function :math:`f(x)` where the variable :math:`x` is normally
# distributed :math:`x \sim p(x) = \mathcal{N}(\mu, \sigma^2)`.
# Then, to evaluate the expectaton of $f$, we can apply the change-of-variables
#
# .. math::
#     z = \frac{x - \mu}{\sqrt{2}\sigma} \Leftrightarrow \sqrt{2}\sigma z + \mu,
#
# and use Gauss-Hermite quadrature, which leads to
#
# .. math::
#
#     \mathbb{E}_{p(x)}[f(x)]
#     & = \int \frac{1}{\sigma \sqrt{2\pi}}
#              \exp \left ( -\frac{(x - \mu)^2}{2\sigma^2} \right ) f(x) dx \\
#     & = \frac{1}{\sqrt{\pi}} \int
#              \exp ( - z^2 ) f(\sqrt{2}\sigma z + \mu) dz \\
#     & \approx \frac{1}{\sqrt{\pi}} \sum_{i=1}^{m} w_i f(\sqrt{2}\sigma z_i + \mu)
#
# where we've used integration by substitution with :math:`dx = \sqrt{2} \sigma dz`.

quadrature_size = 25

# %%

X_samples, weights = np.polynomial.hermite.hermgauss(quadrature_size)

# %%

fig, ax = plt.subplots()

ax.scatter(X_samples, weights)
ax.set_xlabel(r'$x_i$')
ax.set_ylabel(r'$w_i$')

plt.show()

# %%

fig, ax = plt.subplots()

ax.plot(X_pred, h(X_pred), label='$f(x)$')
ax.scatter(X_samples, h(X_samples), c=weights, cmap="Blues")

ax.set_xlabel(r'$x$')
ax.set_ylabel(r'$h(x)$')

plt.show()

# %%


def gauss_hermite_expectation(fn, p, quadrature_size):

    def transform(x, loc, scale):

        return np.sqrt(2) * scale * x + loc

    x, weights = np.polynomial.hermite.hermgauss(quadrature_size)
    y = transform(x, p.loc, p.scale)

    return tf.reduce_sum(weights * fn(y), axis=-1) / tf.sqrt(np.pi)


def divergence_gauss_hermite(q_log_prob_fn, p, quadrature_size,
                             discrepancy_fn=tfp.vi.kl_forward):
    """
    Compute D_f[p || q]
        = E_{q(x)}[f(p(x)/q(x))]
        = E_{p(x)}[r(x)^{-1} f(r(x))]          -- r(x) = p(x)/q(x)
        = E_{p(x)}[exp(-log r(x)) g(log r(x))] -- g(.) = f(exp(.))
        = E_{p(x)}[h(x)]                       -- h(x) = exp(-log r(x)) g(log r(x))
    using Gauss-Hermite quadrature assuming p(x) is Gaussian.
    Note `discrepancy_fn` corresponds to function `g`.
    """
    def log_ratio(x):
        return p.log_prob(x) - q_log_prob_fn(x)

    def h(x):
        return tf.exp(-log_ratio(x)) * discrepancy_fn(log_ratio(x))

    return gauss_hermite_expectation(h, p, quadrature_size)

# %%


divergence_gauss_hermite(q.log_prob, p, quadrature_size).numpy()

# %%
# Comparisons

lst = []

for size in range(1, max_size+1):

    for seed in range(num_seeds):

        kl = divergence_monte_carlo(q.log_prob, p, size, seed=seed).numpy()
        lst.append(dict(kl=kl, size=size, seed=seed, approximation="Monte Carlo"))

    kl = divergence_gauss_hermite(q.log_prob, p, size).numpy()
    lst.append(dict(kl=kl, size=size, seed=0, approximation="Gauss-Hermite"))

df = pd.DataFrame(lst)

# %%
# Results

fig, ax = plt.subplots()

ax.axhline(y=kl_exact, color='tab:red', label='Exact')

sns.lineplot(x='size', y='kl', hue="approximation", data=df, ax=ax)

ax.set_xscale("log")
ax.set_ylabel("KL divergence")

plt.show()
