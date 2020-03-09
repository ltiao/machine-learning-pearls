"""Datasets module."""

import numpy as np
import pandas as pd

import tensorflow as tf

from sklearn.utils import check_random_state
from pathlib import Path

SEED = 42


def synthetic_sinusoidal(x):

    return np.sin(12.0*x) + 0.66*np.cos(25.0*x)


def make_dataset(latent_fn, num_train, num_features, noise_variance,
                 x_min=0., x_max=1., squeeze=True, random_state=SEED):
    """
    Make synthetic dataset.

    Examples
    --------

    Test

    .. plot::
        :context: close-figs

        from etudes.datasets import synthetic_sinusoidal, make_dataset

        num_train = 64 # nbr training points in synthetic dataset
        num_index_points = 256
        num_features = 1
        observation_noise_variance = 1e-1

        f = synthetic_sinusoidal
        X_pred = np.linspace(-0.6, 0.6, num_index_points).reshape(-1, num_features)
        X_train, Y_train = make_dataset(f, num_train, num_features,
                                        observation_noise_variance,
                                        x_min=-0.5, x_max=0.5)

        fig, ax = plt.subplots()

        ax.plot(X_pred, f(X_pred), label="true")
        ax.scatter(X_train, Y_train, marker='x', color='k',
                    label="noisy observations")

        ax.legend()

        ax.set_xlabel(r'$x$')
        ax.set_ylabel(r'$y$')

        plt.show()
    """
    rng = check_random_state(random_state)

    eps = noise_variance * rng.randn(num_train, num_features)

    X = x_min + (x_max - x_min) * rng.rand(num_train, num_features)
    Y = latent_fn(X) + eps

    if squeeze:
        Y = np.squeeze(Y)

    return X, Y


def binarize(positive_label=3, negative_label=5):
    """
    MNIST binary classification.

    Examples
    --------

    .. plot::
        :context: close-figs

        import tensorflow as tf

        from etudes.datasets import binarize
        from etudes.plotting import plot_image_grid

        @binarize(positive_label=2, negative_label=7)
        def binary_mnist_load_data():
            return tf.keras.datasets.mnist.load_data()

        (X_train, Y_train), (X_test, Y_test) = binary_mnist_load_data()

        num_train, img_rows, img_cols = X_train.shape
        num_test, img_rows, img_cols = X_test.shape

        fig, (ax1, ax2) = plt.subplots(ncols=2)

        plot_image_grid(ax1, X_train[Y_train == 0],
                        shape=(img_rows, img_cols), nrows=10, cmap="cividis")

        plot_image_grid(ax2, X_train[Y_train == 1],
                        shape=(img_rows, img_cols), nrows=10, cmap="cividis")

        plt.show()
    """

    # TODO: come up with remote descriptive name
    def d(X, y, label, new_label=1):

        X_val = X[y == label]
        y_val = np.full(len(X_val), new_label)

        return X_val, y_val

    def binarize_decorator(load_data_fn):

        def new_load_data_fn():

            (X_train, Y_train), (X_test, Y_test) = load_data_fn()

            X_train_pos, Y_train_pos = d(X_train, Y_train,
                                         label=positive_label, new_label=1)
            X_train_neg, Y_train_neg = d(X_train, Y_train,
                                         label=negative_label, new_label=0)

            X_train_new = np.vstack([X_train_pos, X_train_neg])
            Y_train_new = np.hstack([Y_train_pos, Y_train_neg])

            X_test_pos, Y_test_pos = d(X_test, Y_test,
                                       label=positive_label, new_label=1)
            X_test_neg, Y_test_neg = d(X_test, Y_test,
                                       label=negative_label, new_label=0)

            X_test_new = np.vstack([X_test_pos, X_test_neg])
            Y_test_new = np.hstack([Y_test_pos, Y_test_neg])

            return (X_train_new, Y_train_new), (X_test_new, Y_test_new)

        return new_load_data_fn

    return binarize_decorator


@binarize(positive_label=2, negative_label=7)
def binary_mnist_load_data():
    return tf.keras.datasets.mnist.load_data()


def coal_mining_disasters_load_data(base_dir="../datasets/"):
    """
    Coal mining disasters dataset.

    Examples
    --------

    .. plot::
        :context: close-figs

        from etudes.datasets import coal_mining_disasters_load_data

        X, y = coal_mining_disasters_load_data()

        fig, ax = plt.subplots()

        ax.vlines(X.squeeze(), ymin=0, ymax=y, linewidth=0.5, alpha=0.8)

        ax.set_xlabel("days")
        ax.set_ylabel("incidents")

        plt.show()
    """
    base = Path(base_dir).joinpath("coal-mining-disasters")

    data = pd.read_csv(base / "data.csv", names=["count", "days"], header=None)

    X = np.expand_dims(data["days"].values, axis=-1)
    y = data["count"].values

    return X, y


def mauna_loa_load_dataframe(base_dir="../datasets/"):
    """
    Mauna Loa dataset.

    Examples
    --------

    .. plot::
        :context: close-figs

        import seaborn as sns
        from etudes.datasets import mauna_loa_load_dataframe

        data = mauna_loa_load_dataframe()

        g = sns.relplot(x='date', y='average', kind="line",
                        data=data, height=5, aspect=1.5, alpha=0.8)
        g.set_ylabels(r"average $\mathrm{CO}_2$ (ppm)")
    """
    base = Path(base_dir).joinpath("mauna-loa-co2")

    column_names = ["year", "month", "date", "average", "interpolated",
                    "trend", "num_days"]

    data = pd.read_csv(base / "co2_mm_mlo.txt", names=column_names,
                       comment="#", header=None, sep=r"\s+")
    data = data[data.average > 0]

    return data
