====================
Posterior Predictive
====================

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
    from pearls.gaussian_processes import gp_sample_custom, dataframe_from_gp_samples, dataframe_from_gp_summary

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

.. plot::
   :context: close-figs
   :include-source:

    n_train = 20 # nbr training points in synthetic dataset
    observation_noise_variance = 1e-1

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

    with tf.Session() as sess:
        _nll = sess.run(nll)

.. plot::
   :context: close-figs
   :include-source:

    fig, ax = plt.subplots(subplot_kw=dict(projection="3d", azim=25, elev=35))

    with tf.Session() as sess:

        ax.plot_surface(amplitude, length_scale, _nll,
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

    contours = ax.contour(_amplitude, _length_scale, _nll, cmap="Spectral_r")

    fig.colorbar(contours, ax=ax)
    ax.clabel(contours, fmt='%.1f')

    ax.set_xlabel(r"amplitude $\sigma$")
    ax.set_ylabel(r"lengthscale $\ell$")

    plt.show()

.. plot::
   :context: close-figs
   :include-source:

    _amplitude, _length_scale = np.broadcast_arrays(amplitude, length_scale)

    amplitude, length_scale = amplitude[10:50:8], length_scale[...,5:45:8]
    kernel = kernel_cls(amplitude=amplitude, length_scale=length_scale)

    theta = np.dstack(np.broadcast_arrays(amplitude, length_scale)).reshape(-1, 2)

    fig, ax = plt.subplots()

    ax.scatter(*theta.T, color='k', marker='x')

    with tf.Session() as sess:

    #     ax.plot_surface(amplitude, length_scale, sess.run(neg_log_likelihood), cmap="coolwarm")
        contours = ax.contour(_amplitude, _length_scale, sess.run(nll), cmap="Spectral_r")

    fig.colorbar(contours, ax=ax)
    ax.clabel(contours, fmt='%.1f')

    ax.set_xlabel(r"lengthscale $\ell$")
    ax.set_ylabel(r"amplitude $\sigma$")

    plt.show()

.. plot::
   :context: close-figs
   :include-source:

    gprm = tfd.GaussianProcessRegressionModel(
        kernel=kernel, index_points=X_q, observation_index_points=X, observations=Y,
        observation_noise_variance=observation_noise_variance, jitter=jitter
    )


.. plot::
   :context: close-figs
   :include-source:

    gprm_mean = gprm.mean()
    gprm_stddev = gprm.stddev()

    with tf.Session() as sess:
        gprm_mean_arr, gprm_stddev_arr = sess.run([gprm_mean, gprm_stddev])

    data = dataframe_from_gp_summary(gprm_mean_arr, gprm_stddev_arr, amplitude,
                                     length_scale, X_q)

    def scatterplot(X, Y, ax=None, *args, **kwargs):

        if ax is None:
            ax = plt.gca()

        ax.scatter(X, Y, s=8.0**2, marker='x', color='k')

    def fill_between(index_points, mean, stddev, ax=None, **kwargs):

        if ax is None:
            ax = plt.gca()

        ax.fill_between(index_points, mean - 2*stddev, mean + 2*stddev, **kwargs)

    g = sns.relplot(x="index_point", y="mean",
                    row="amplitude", row_order=amplitude[::-1].squeeze(), col="length_scale", 
                    height=5.0, aspect=1.2, kind="line", data=data, alpha=0.7, linewidth=3.0)
    g.map(scatterplot, X=X, Y=Y)
    g.map(fill_between, "index_point", "mean", "stddev", alpha=0.1)
    g.set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
                 col_template=r"lengthscale $\ell={{{col_name:.3f}}}$")
    g.set_axis_labels(r"$x$", r"$m(x)$")
    g.set(ylim=(-2.0, 1.5))

.. plot::
   :context: close-figs
   :include-source:

    gp_samples = gp_sample_custom(gprm, n_samples, seed=seed)

    with tf.Session() as sess:
        gp_samples_arr = sess.run(gp_samples)

    data = dataframe_from_gp_samples(gp_samples_arr, X_q, 
                                     amplitude, length_scale,
                                     n_samples)

    g = sns.relplot(x="index_point", y="function_value", hue="sample",
                    row="amplitude", row_order=amplitude[::-1].squeeze(), col="length_scale", 
                    height=5.0, aspect=1.2, kind="line", data=data, alpha=0.7, linewidth=2.0)
    g.map(scatterplot, X=X, Y=Y)
    g.set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
                 col_template=r"lengthscale $\ell={{{col_name:.3f}}}$")
    g.set_axis_labels(r"$x$", r"$f^{(i)}(x)$")
    g.set(ylim=(-2.0, 1.5))

.. plot::
   :context: close-figs
   :include-source:

    # instantiate Gaussian Process
    gp = tfd.GaussianProcess(kernel=kernel, index_points=X_q, jitter=jitter)
    gp_samples = gp_sample_custom(gp, n_samples, seed=seed)

    with tf.Session() as sess:
        gp_samples_arr = sess.run(gp_samples)

    data = dataframe_from_gp_samples(gp_samples_arr, X_q, amplitude,
                                     length_scale, n_samples)

.. plot::
   :context: close-figs
   :include-source:

    g = sns.relplot(x="index_point", y="function_value", hue="sample",
                    row="amplitude", col="length_scale", height=5.0, aspect=1.2,
                    kind="line", data=data, alpha=0.7, linewidth=3.0)
    g.set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
                 col_template=r"lengthscale $\ell={{{col_name:.3f}}}$")
    g.set_axis_labels(r"$x$", r"$f^{(i)}(x)$")