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
    n_index_points = 256 # nbr of index points
    n_samples = 5

    seed = 23 # set random seed for reproducibility
    random_state = np.random.RandomState(seed)

    X_q = np.linspace(-1.0, 1.0, n_index_points).reshape(-1, n_features)

    amplitude, length_scale = np.ogrid[0.05:0.16:0.05, 0.025:0.16:0.05]

    kernel = kernels.ExponentiatedQuadratic(amplitude=amplitude, 
                                            length_scale=length_scale)

    gp = tfd.GaussianProcess(kernel=kernel, index_points=X_q)
    gp_samples = gp.sample(n_samples, seed=seed)

    ind = pd.MultiIndex.from_product([list(map("$i={}$".format, range(n_samples))),
                                      amplitude.squeeze(),
                                      length_scale.squeeze(),
                                      X_q.squeeze()], 
                                     names=["sample", "amplitude", "length_scale", "$x$"])

    with tf.Session() as sess:
        d = pd.DataFrame(gp_samples.eval().ravel(), 
                         index=ind, columns=["$f^{(i)}(x)$"])

    g = sns.relplot(x="$x$", y="$f^{(i)}(x)$", hue="sample",
                    row="amplitude", col="length_scale",
                    height=5.0, aspect=1.0, 
                    kind="line", data=d.reset_index())
    g.set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
                 col_template=r"lengthscale $\ell={{{col_name:.3f}}}$")
