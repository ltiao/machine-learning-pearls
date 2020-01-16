import numpy as np

from pathlib import Path


def load_zalando(base_dir="datasets/zalando-phd-interview"):

    p = Path(base_dir)

    X = np.load(p / "features.npy")
    y = np.load(p / "labels.npy")

    return X, y
