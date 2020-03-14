import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


def expectation_gauss_hermite(fn, normal, quadrature_size):

    def transform(x, loc, scale):

        return np.sqrt(2) * scale * x + loc

    x, weights = np.polynomial.hermite.hermgauss(quadrature_size)
    y = transform(x, normal.loc, normal.scale)

    return tf.reduce_sum(weights * fn(y), axis=-1) / tf.sqrt(np.pi)


def divergence_gauss_hermite(p, q, quadrature_size, under_p=True,
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
        return p.log_prob(x) - q.log_prob(x)

    if under_p:

        # TODO: Raise exception if `p` is non-Gaussian.
        w = lambda x: tf.exp(-log_ratio(x))
        normal = p

    else:

        # TODO: Raise exception if `q` is non-Gaussian.
        w = lambda x: 1.0
        normal = q

    def fn(x):
        return w(x) * discrepancy_fn(log_ratio(x))

    return expectation_gauss_hermite(fn, normal, quadrature_size)
