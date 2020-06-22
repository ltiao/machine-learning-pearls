# -*- coding: utf-8 -*-
"""
Perlin noise
============

Hello world
"""
# sphinx_gallery_thumbnail_number = 4

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.special import expit
from noise import pnoise2

h, w = 200, 200

factor = 10.0

num_steps = 50
step_size = 1.0

max_size = step_size * num_steps

y, x = np.ogrid[:h, :w]
X, Y = np.broadcast_arrays(x, y)

Z = expit(factor * np.vectorize(pnoise2)(x/w, y/h, octaves=6))  # range [-1, 1]
theta = np.pi * Z  # range [-pi, pi]

# %%
# Foo
# ---
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_title(r"angle $\theta$ (rad)")

contours = ax.pcolormesh(X, Y, theta, cmap="twilight")
fig.colorbar(contours, ax=ax)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()

# %%
# Bar
# ---

hr, wr = 10, 10

ydown = y[::hr]
xdown = x[..., ::wr]
theta_down = theta[::hr, ::wr]
Xdown = X[::hr, ::wr]
Ydown = Y[::hr, ::wr]

data = pd.DataFrame(theta_down / np.pi,
                    index=ydown.squeeze(axis=1),
                    columns=xdown.squeeze(axis=0))

dx = step_size * np.cos(theta_down)
dy = step_size * np.sin(theta_down)

# %%
# Baz
# ---
fig, ax = plt.subplots(figsize=(10, 8))

ax.set_title(r"angle $\theta / \pi$ (rad)")

sns.heatmap(data, annot=True, fmt=".2f", cmap="twilight", ax=ax)
ax.invert_yaxis()

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()

# %%
# Vector field
# ------------
fig, ax = plt.subplots(figsize=(10, 8))

ax.quiver(xdown, ydown, dx, dy)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()

# %%
# Path
# ----

num_lines = 5000

lines_x = []
lines_y = []

for l in range(num_lines):

    x = w * np.random.rand()
    y = h * np.random.rand()

    xs = [x]
    ys = [y]

    for step in range(num_steps):

        j, i = int(x), int(y)

        if not (0 <= i < w and 0 <= j < h):
            break

        x += step_size * np.cos(theta[i, j])
        y += step_size * np.sin(theta[i, j])

        xs.append(x)
        ys.append(y)

    lines_x.append(xs)
    lines_y.append(ys)

fig, ax = plt.subplots(figsize=(10, 8))

for xs, ys in zip(lines_x, lines_y):

    # ax.plot(xs, ys, marker='o', linestyle="none", mfc="none")
    ax.plot(xs, ys)

ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$")

plt.show()
