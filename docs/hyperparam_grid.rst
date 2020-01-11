====
Grid
====

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
    n_samples = 5 # nbr of GP prior samples 
    jitter = 1e-15

    seed = 42 # set random seed for reproducibility
    random_state = np.random.RandomState(seed)

    X_q = np.linspace(-1.0, 1.0, n_index_points).reshape(-1, n_features)

    amplitude, length_scale_inv = np.ogrid[0.05:0.16:0.05, 10.0:0.5:-1.25]

    length_scale = 1.0 / length_scale_inv

    kernel = kernels.RationalQuadratic(amplitude=amplitude, 
                                       length_scale=length_scale)

    gp = tfd.GaussianProcess(kernel=kernel, index_points=X_q, jitter=jitter)

    def gp_sample_custom(gp, n_samples, seed=None):

        gp_marginal = gp.get_marginal_distribution()

        batch_shape = tf.ones(gp_marginal.batch_shape.rank, dtype=tf.int32)
        event_shape = gp_marginal.event_shape

        sample_shape = tf.concat([[n_samples], batch_shape, event_shape], axis=0)

        base_samples = gp_marginal.distribution.sample(sample_shape, seed=seed)
        gp_samples = gp_marginal.bijector.forward(base_samples)

        return gp_samples

    gp_samples = gp_sample_custom(gp, n_samples, seed=seed)

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
                    height=5.0, aspect=1.0, legend=False,
                    kind="line", data=d.reset_index(), linewidth=3.0)
                    # facet_kws=dict(margin_titles=True))
    g.set_titles(template='')
           # .set_titles(row_template=r"amplitude $\sigma={{{row_name:.2f}}}$",
           #             col_template=r"lengthscale $\ell={{{col_name:.3f}}}$")
    g.despine(top=True, right=True, left=True, bottom=True, trim=False)
    g.set(xticks=[], yticks=[])
    g.set_axis_labels("", "")
    g.fig.subplots_adjust(wspace=.01, hspace=.01)