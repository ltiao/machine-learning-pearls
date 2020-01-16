"""Console script for pearls."""
import sys
import click

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)

from .datasets import load_zalando


@click.command()
def main(args=None):
    """Console script for pearls."""

    X, y = load_zalando()

    print(X.shape, y.shape)

    click.echo("Replace this message by putting your code into "
               "pearls.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
