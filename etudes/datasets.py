"""Datasets module."""

import os.path
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sps
import pickle as pkl

import tensorflow as tf
import tensorflow_probability as tfp

from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import check_random_state, shuffle as _shuffle
from pathlib import Path

from functools import partial


SEED = 42

tfd = tfp.distributions


def synthetic_sinusoidal(x):

    return np.sin(12.0*x) + 0.66*np.cos(25.0*x)


def make_regression_dataset(latent_fn=synthetic_sinusoidal):
    """
    Make synthetic dataset.

    Examples
    --------

    Test

    .. plot::
        :context: close-figs

        from etudes.datasets import synthetic_sinusoidal, make_regression_dataset

        num_train = 64 # nbr training points in synthetic dataset
        num_index_points = 256
        num_features = 1
        observation_noise_variance = 1e-1

        f = synthetic_sinusoidal
        X_pred = np.linspace(-0.6, 0.6, num_index_points).reshape(-1, num_features)

        load_data = make_regression_dataset(f)
        X_train, Y_train = load_data(num_train, num_features,
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

    def load_data(num_samples, num_features, noise_variance,
                  x_min=0., x_max=1., squeeze=True, random_state=SEED):

        rng = check_random_state(random_state)

        eps = noise_variance * rng.randn(num_samples, num_features)

        X = x_min + (x_max - x_min) * rng.rand(num_samples, num_features)
        Y = latent_fn(X) + eps

        if squeeze:
            Y = np.squeeze(Y)

        return X, Y

    return load_data


def make_classification_dataset(X_pos, X_neg, shuffle=False, dtype="float64",
                                random_state=None):

    X = np.vstack([X_pos, X_neg]).astype(dtype)
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    if shuffle:
        X, y = _shuffle(X, y, random_state=random_state)

    return X, y


def make_density_ratio_estimation_dataset(p=None, q=None):

    if p is None:
        p = tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=[0.3, 0.7]),
            components_distribution=tfd.Normal(loc=[2.0, -3.0],
                                               scale=[1.0, 0.5]))

    if q is None:
        q = tfd.Normal(loc=0.0, scale=2.0)

    def load_data(num_samples, rate=0.5, dtype="float64", seed=SEED):

        num_p = int(num_samples * rate)
        num_q = num_samples - num_p

        X_p = p.sample(sample_shape=(num_p, 1), seed=seed).numpy()
        X_q = q.sample(sample_shape=(num_q, 1), seed=seed).numpy()

        X, y = make_classification_dataset(X_p, X_q, dtype=dtype,
                                           random_state=seed)

        return X, y

    return load_data


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


def get_sequence_path(sequence_num, base_dir="../datasets"):

    return Path(base_dir).joinpath("bee-dance", "zips", "data",
                                   f"sequence{sequence_num:d}", "btf")


bee_dance_filenames = dict(
    x="ximage.btf",
    y="yimage.btf",
    t="timage.btf",
    label="label0.btf",
    timestamp="timestamp.btf"
)


def read_sequence_column(sequence_num, col_name, base_dir="../datasets"):

    sequence_path = get_sequence_path(sequence_num, base_dir=base_dir)

    return pd.read_csv(sequence_path / bee_dance_filenames[col_name],
                       names=[col_name], header=None)


def read_sequence(sequence_num, base_dir="../datasets"):

    left = None

    for col_name in bee_dance_filenames:

        right = read_sequence_column(sequence_num, col_name, base_dir=base_dir)

        if left is None:
            left = right
        else:
            left = pd.merge(left, right, left_index=True, right_index=True)

    change_point = left.label != left.label.shift()
    phase = change_point.cumsum()

    return left.assign(change_point=change_point, phase=phase)


def load_bee_dance_dataframe(base_dir="../datasets"):

    sequences = []

    for i in range(6):

        sequence_num = i + 1
        sequence = read_sequence(sequence_num, base_dir=base_dir) \
            .assign(sequence=sequence_num)
        sequences.append(sequence)

    return pd.concat(sequences, axis="index")


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


def load_cora(data_home="datasets/legacy/cora"):

    base = Path(data_home)

    df = pd.read_csv(base.joinpath("cora.content"),
                     sep=r"\s+", header=None, index_col=0)

    features_df = df.iloc[:, :-1]
    labels_df = df.iloc[:, -1]

    X_all = features_df.values

    y_all = LabelBinarizer().fit_transform(labels_df.values)

    edge_list_df = pd.read_csv(base.joinpath("cora.cites"),
                               sep=r"\s+", header=None)

    idx_map = {j: i for i, j in enumerate(df.index)}

    H = nx.from_pandas_edgelist(edge_list_df, 0, 1)
    G = nx.relabel.relabel_nodes(H, idx_map)

    A = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G.nodes()), format='coo')

    return (X_all, y_all, A)


def load_pickle(name, ext, data_home="datasets", encoding='latin1'):

    path = os.path.join(data_home, name, "ind.{0}.{1}".format(name, ext))

    with open(path, "rb") as f:

        return pkl.load(f, encoding=encoding)


def load_test_indices(name, data_home="datasets"):

    indices_df = pd.read_csv(os.path.join(data_home, name, "ind.{0}.test.index".format(name)), header=None)
    indices = indices_df.values.squeeze()

    return indices


def load_dataset(name, data_home="datasets"):

    exts = ['tx', 'ty', 'allx', 'ally', 'graph']

    (X_test,
     y_test,
     X_rest,
     y_rest,
     G_dict) = map(partial(load_pickle, name, data_home=data_home), exts)

    _, D = X_test.shape
    _, K = y_test.shape

    ind_test_perm = load_test_indices(name, data_home)
    ind_test = np.sort(ind_test_perm)

    num_test = len(ind_test)
    num_test_full = ind_test[-1] - ind_test[0] + 1

    # TODO: Issue warning if `num_isolated` is non-zero.
    num_isolated = num_test_full - num_test

    # normalized zero-based indices
    ind_test_norm = ind_test - np.min(ind_test)

    # features
    X_test_full = sps.lil_matrix((num_test_full, D))
    X_test_full[ind_test_norm] = X_test

    X_all = sps.vstack((X_rest, X_test_full)).toarray()
    X_all[ind_test_perm] = X_all[ind_test]

    # targets
    y_test_full = np.zeros((num_test_full, K))
    y_test_full[ind_test_norm] = y_test

    y_all = np.vstack((y_rest, y_test_full))
    y_all[ind_test_perm] = y_all[ind_test]

    # graph
    G = nx.from_dict_of_lists(G_dict)
    A = nx.to_scipy_sparse_matrix(G, format='coo')

    return (X_all, y_all, A)
