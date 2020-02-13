"""Main module."""

import tensorflow as tf
import pandas as pd


def gp_sample_custom(gp, n_samples, seed=None):

    gp_marginal = gp.get_marginal_distribution()

    batch_shape = tf.ones(gp_marginal.batch_shape.rank, dtype=tf.int32)
    event_shape = gp_marginal.event_shape

    sample_shape = tf.concat([[n_samples], batch_shape, event_shape], axis=0)

    base_samples = gp_marginal.distribution.sample(sample_shape, seed=seed)
    gp_samples = gp_marginal.bijector.forward(base_samples)

    return gp_samples


def dataframe_from_gp_samples(gp_samples_arr, X_q, amplitude, length_scale,
                              n_samples):

    names = ["sample", "amplitude", "length_scale", "index_point"]

    v = [list(map(r"$i={}$".format, range(n_samples))),
         amplitude.squeeze(), length_scale.squeeze(), X_q.squeeze()]

    index = pd.MultiIndex.from_product(v, names=names)

    d = pd.DataFrame(gp_samples_arr.ravel(), index=index, columns=["function_value"])

    return d.reset_index()


def dataframe_from_gp_summary(gp_mean_arr, gp_stddev_arr, amplitude,
                              length_scale, index_point):

    names = ["amplitude", "length_scale", "index_point"]
    v = [amplitude.squeeze(), length_scale.squeeze(), index_point.squeeze()]

    index = pd.MultiIndex.from_product(v, names=names)

    d1 = pd.DataFrame(gp_mean_arr.ravel(), index=index, columns=["mean"])
    d2 = pd.DataFrame(gp_stddev_arr.ravel(), index=index, columns=["stddev"])

    data = pd.merge(d1, d2, left_index=True, right_index=True)

    return data.reset_index()
