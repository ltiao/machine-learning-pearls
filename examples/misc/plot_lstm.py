# -*- coding: utf-8 -*-
"""
Long short-term memory (LSTM) networks
======================================

Hello world
"""
# sphinx_gallery_thumbnail_number = 4

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, RepeatVector, TimeDistributed
# %%

num_seqs = 5
seq_len = 25
num_features = 1

seed = 42  # set random seed for reproducibility
random_state = np.random.RandomState(seed)
# %%
# Many-to-many LSTM
# -----------------

# generate random walks in Euclidean space
inputs = np.cumsum(random_state.randn(num_seqs, seq_len, num_features), axis=1)
lstm = LSTM(units=1, return_sequences=True)
output = lstm(inputs)
# %%

print(output.shape)
# %%

fig, ax = plt.subplots()

ax.plot(inputs[..., 0].T)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$x(t)$")

plt.show()
# %%

fig, ax = plt.subplots()

ax.plot(output.numpy()[..., 0].T)

ax.set_xlabel(r"$t$")
ax.set_ylabel(r"$h(t)$")

plt.show()
# %%

fig, ax = plt.subplots()

ax.plot(inputs[..., 0].T,
        output.numpy()[..., 0].T)

ax.set_xlabel(r"$x(t)$")
ax.set_ylabel(r"$h(t)$")

plt.show()

# %%
# One-to-many LSTM
# ----------------
seq_len = 100
num_index_points = 128
xmin, xmax = -10.0, 10.0
X_grid = np.linspace(xmin, xmax, num_index_points).reshape(-1, num_features)
# %%

model = Sequential([
    RepeatVector(seq_len, input_shape=(num_features,)),
    Bidirectional(LSTM(units=32, return_sequences=True), merge_mode="concat"),
    TimeDistributed(Dense(1))
])
model.summary()
# %%

Z = model(X_grid)
Z
# %%

T_grid = np.arange(seq_len)

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

ax.plot_wireframe(T_grid, X_grid, Z.numpy().squeeze(axis=-1), alpha=0.6)
# ax.plot_surface(T_grid, X_grid, Z.numpy().squeeze(axis=-1),
#                 edgecolor='k', linewidth=0.5, cmap="Spectral_r")

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$x$')
ax.set_zlabel(r"$h(x, t)$")

plt.show()
