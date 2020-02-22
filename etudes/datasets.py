"""Datasets module."""

import numpy as np

from sklearn.utils import check_random_state


def synthetic_sinusoidal(x):

    return np.sin(12.0*x) + 0.66*np.cos(25.0*x)


def make_dataset(latent_fn, n_train, n_features, noise_variance,
                 x_min=0., x_max=1., squeeze=True, random_state=None):

    rng = check_random_state(random_state)

    eps = noise_variance * rng.randn(n_train, n_features)

    X = x_min + (x_max - x_min) * rng.rand(n_train, n_features)
    Y = latent_fn(X) + eps

    if squeeze:
        Y = np.squeeze(Y)

    return X, Y
