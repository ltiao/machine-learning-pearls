"""Plotting module."""

import numpy as np


def plot_image_grid(ax, images, shape, nrows=20, ncols=None, cmap=None):

    if ncols is None:
        ncols = nrows

    grid = images[:nrows*ncols].reshape(nrows, ncols, *shape).squeeze()

    return ax.imshow(np.vstack(np.dstack(grid)), cmap=cmap)
