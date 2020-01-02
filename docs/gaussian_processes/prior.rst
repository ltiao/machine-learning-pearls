================
Gaussian Process
================

.. plot::
   :context: close-figs
   :include-source:

    import numpy as np

    import tensorflow as tf
    import tensorflow_probability as tfp

    import matplotlib.pyplot as plt
    import seaborn as sns

.. plot::
   :context: close-figs
   :include-source:

    # shortcuts
    tfd = tfp.distributions
    kernels = tfp.math.psd_kernels

    # constants
    n_features = 1 # dimensionality
    n_train = 12 # nbr training points in synthetic dataset
    n_index_points = 128 # nbr of index points

    seed = 23 # set random seed for reproducibility
    random_state = np.random.RandomState(seed)

A Gaussian process model is specified in terms of a mean function :math:`m(\mathbf{x})`
and kernel or covariance function :math:`\kappa(\mathbf{x}, \mathbf{x}')`. 
The Gaussian process prior on regression functions is denoted by

.. math::

    f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), \kappa(\mathbf{x}, \mathbf{x}'))

It is common to simply specify :math:`m(x)=0` since the GP is flexible enough 
to also model the mean arbitrarily well, so let's go ahead and specify a 
covariance function.

.. plot::
   :context: close-figs
   :include-source:

    k = kernels.ExponentiatedQuadratic()

.. plot::
   :context: close-figs
   :include-source:

    index_points = np.linspace(-5.0, 5.0, n_index_points).reshape(-1, 1)

    fig, ax = plt.subplots()

    with tf.Session() as sess:
        ax.plot(index_points, sess.run(k.apply(index_points, np.zeros(1))))

    ax.set_xlabel('$x$')
    ax.set_ylabel('$k(x, 0)$')
    # ax.set_title('RBF Kernel ($\ell={{{[0]:.2f}}}$)'
    #              .format(k.lengthscales.value))

    plt.show()

Prior Samples
=============


.. plot::
   :context: close-figs
   :include-source:

    gp = tfd.GaussianProcess(k, index_points)

    samples = gp.sample(6, seed=seed)

    fig, ax = plt.subplots()

    with tf.Session() as sess:
        ax.plot(index_points, samples.eval().T)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$f(x)$')
    ax.set_title('Draws of $f(x)$ from GP prior')

    plt.show()


Synthetic Dataset
=================

.. plot::
   :context: close-figs
   :include-source:

    f = lambda x: np.sin(12.0*x) + 0.66*np.cos(25.0*x) + 3.0
    
    X = random_state.rand(n_train, n_features)
    eps = 0.1*random_state.randn(n_train, n_features)
    Y = f(X) + eps

    fig, ax = plt.subplots()

    ax.scatter(X, Y, marker='x', color='k')

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')

    plt.show()