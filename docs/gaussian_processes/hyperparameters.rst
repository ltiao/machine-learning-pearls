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
    import pandas as pd

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
    n_samples = 8

    seed = 23 # set random seed for reproducibility
    random_state = np.random.RandomState(seed)

    X_q = np.linspace(-5.0, 5.0, n_index_points).reshape(-1, 1)

.. plot::
   :context: close-figs
   :include-source:

    amplitude, length_scale = np.ogrid[0.05:0.2:0.05, 0.1:2.1:0.7]

    kernel = kernels.ExponentiatedQuadratic(amplitude=amplitude, 
                                            length_scale=length_scale)

    gp = tfd.GaussianProcess(kernel, X_q)
    gp_samples = gp.sample(n_samples, seed=seed)

    ind = pd.MultiIndex.from_product([range(n_samples), amplitude.squeeze(), length_scale.squeeze(), X_q.squeeze()], 
                                     names=["sample", "amplitude", "length_scale", "$x$"])

    with tf.Session() as sess:
        d = pd.DataFrame(gp_samples.eval().ravel(), index=ind, columns=["$f(x)$"])

    g = sns.relplot(x="$x$", y="$f(x)$", hue="sample",
                    row="amplitude", col="length_scale",
                    height=8, aspect=golden_ratio,
                    kind="line", data=d.reset_index())

.. TODO: 
.. - Fix hue to be categorical
.. - Fix subplot heading floating point precision
.. - Change amplitude and lengthscale column names to Greek symbols
