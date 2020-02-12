===================
Marginal Likelihood
===================

.. plot::
   :context: close-figs
   :include-source:

    import numpy as np

    import tensorflow as tf
    import tensorflow_probability as tfp

    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from mpl_toolkits.mplot3d import Axes3D
    from pearls.gaussian_processes import gp_sample_custom, dataframe_from_gp_samples

    # shortcuts
    tfd = tfp.distributions
    kernels = tfp.math.psd_kernels

    # constants
    n_features = 1 # dimensionality
    n_index_points = 256 # nbr of index points
    n_samples = 5 # nbr of GP prior samples 
    jitter = 1e-6
    kernel_cls = kernels.ExponentiatedQuadratic

    seed = 42 # set random seed for reproducibility
    random_state = np.random.RandomState(seed)

    # index points
    X_q = np.linspace(-1.0, 1.0, n_index_points).reshape(-1, n_features)

    # kernel specification
    amplitude, length_scale = np.ogrid[5e-2:4.0:100j, 1e-5:5e-1:100j]
    kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)

Synthetic Dataset
=================

.. plot::
   :context: close-figs
   :include-source:

    n_train = 12 # nbr training points in synthetic dataset
    observation_noise_variance = 0.1

    f = lambda x: np.sin(12.0*x) + 0.66*np.cos(25.0*x)

    X = random_state.rand(n_train, n_features) - 0.5
    eps = observation_noise_variance * random_state.randn(n_train, n_features)
    Y = np.squeeze(f(X) + eps)

    fig, ax = plt.subplots()

    ax.plot(X_q, f(X_q), label="true")
    ax.scatter(X, Y, marker='x', color='k', label="noisy observations")

    ax.legend()

    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    plt.show()

.. plot::
   :context: close-figs
   :include-source:

    gp = tfd.GaussianProcess(
        kernel=kernel,
        index_points=X,
        observation_noise_variance=observation_noise_variance
    )
    nll = - gp.log_prob(Y)

.. plot::
   :context: close-figs
   :include-source:

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d", azim=25, elev=35))

    with tf.Session() as sess:

        ax.plot_surface(amplitude, length_scale, sess.run(nll), 
                        rstride=1, cstride=1, edgecolor='none', cmap="Spectral_r")

    ax.set_xlabel(r"amplitude $\sigma$")
    ax.set_ylabel(r"lengthscale $\ell$")
    ax.set_zlabel("nll")

    plt.show()

.. plot::
   :context: close-figs
   :include-source:

    _amplitude, _length_scale = np.broadcast_arrays(amplitude, length_scale)

    fig, ax = plt.subplots()

    with tf.Session() as sess:

        contours = ax.contour(_amplitude, _length_scale, sess.run(nll), cmap="Spectral_r")

    fig.colorbar(contours, ax=ax)
    ax.clabel(contours, fmt='%.1f')

    ax.set_xlabel(r"amplitude $\sigma$")
    ax.set_ylabel(r"lengthscale $\ell$")

    plt.show()