"""Console script for pearls."""
import sys
import click

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from pearls.datasets import load_zalando

from sklearn.preprocessing import normalize, scale
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                              GradientBoostingClassifier)

TEST_SIZE = 0.4
SEED = None

classifier_names = [
    # "Naive Bayes",
    # "Nearest Neighbors",
    # "Logistic Regression",
    "Neural Net",
    # "Linear SVM",
    # "RBF SVM",
    # "Random Forest",
    # "AdaBoost"
]

classifiers = [
    # GaussianNB(),
    # KNeighborsClassifier(),
    # LogisticRegression(),
    MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=200, verbose=True),
    # SVC(kernel="linear", C=0.025),
    # SVC(gamma=2, C=1),
    # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    # AdaBoostClassifier(),
]


@click.command()
@click.option("--test-size", default=TEST_SIZE, type=click.FLOAT, help="Test set size.")
@click.option("-s", "--seed", default=SEED, type=click.INT, help="Random seed")
def main(test_size, seed):

    X, y = load_zalando()
    # X = scale(X)
    X = normalize(X, norm='l1', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=seed)

    for clf_name, clf in zip(classifier_names, classifiers):
        print(clf_name, clf)
        print(cross_val_score(clf, X_train, y_train, cv=2))
        print()

    click.echo("Replace this message by putting your code into "
               "pearls.cli.main")
    click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
