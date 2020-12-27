# -*- coding: utf-8 -*-
"""
Cryptocurrency Exchange Orderbook
=================================

A snapshot of an orderbook on Binance, a popular cryptocurrency exchange,
for a specified symbol (trading pair).

- Calls the Binance API endpoint using ``requests``.
- Visualizes the result using ``seaborn``.
"""
# sphinx_gallery_thumbnail_number = 1

import requests
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
# %%

pair = ("ETH", "AUD")
symbol = "".join(pair)
binwidth = 5.0

# %%
r = requests.get("https://api.binance.com/api/v3/depth", params=dict(symbol=symbol))
results = r.json()
results
# %%

frames = []
for kind in ["bids", "asks"]:
    frame = pd.DataFrame(data=results[kind], 
                         columns=["price", "amount"],
                         dtype=float).assign(kind=kind)
    frames.append(frame)
# %%
data = pd.concat(frames, axis="index", sort=True)
data
# %%
fig, ax = plt.subplots()

sns.scatterplot(x='amount', y='price', hue='kind', data=data, ax=ax)

ax.set_xlabel(f"Amount ({pair[0]})")
ax.set_ylabel(f"Price ({pair[1]})")

ax.set_xscale("log")

plt.show()
# %%
fig, ax = plt.subplots()

sns.histplot(x='price', hue='kind', binwidth=binwidth, data=data, ax=ax)

ax.set_xlabel(f"Price ({pair[1]})")

plt.show()

# %%
# Weighted by amount.
fig, ax = plt.subplots()

sns.histplot(x='price', weights='amount', hue='kind', binwidth=binwidth,
             data=data, ax=ax)

ax.set_xlabel(f"Price ({pair[1]})")
ax.set_ylabel(f"Amount ({pair[0]})")

plt.show()
