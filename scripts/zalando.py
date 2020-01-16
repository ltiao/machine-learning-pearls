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


from functools import partial

import os.path
import datetime

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

binary_crossentropy_from_logits = partial(binary_crossentropy, from_logits=True)

TEST_SIZE = 0.4
SEED = None
EPOCHS = 50
BATCH_SIZE = 64
L1_FACTOR = 0.0
L2_FACTOR = 5e-5

INITIAL_EPOCH = 0

CHECKPOINT_DIR = "checkpoints/"
CHECKPOINT_PERIOD = 1

SUMMARY_DIR = "logs/"


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


def build_model(output_dim, num_layers=2, num_units=64, activation="relu",
                *args, **kwargs):

    model = Sequential()

    for _ in range(num_layers):

        model.add(Dense(num_units, activation=activation, *args, **kwargs))

    model.add(Dense(output_dim, activation='sigmoid'))

    return model

@click.command()
@click.argument("name")
@click.option("--initial-epoch", default=INITIAL_EPOCH, type=click.INT, 
              help="Epoch at which to resume a previous training run")
@click.option("-e", "--epochs", default=EPOCHS, type=click.INT, help="Number of epochs.")
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=click.INT, help="Batch size.")
@click.option("--l1-factor", default=L1_FACTOR, type=click.FLOAT, help="L1 regularization factor.")
@click.option("--l2-factor", default=L2_FACTOR, type=click.FLOAT, help="L2 regularization factor.")
@click.option("--test-size", default=TEST_SIZE, type=click.FLOAT, help="Test set size.")
@click.option("--checkpoint-dir", default=CHECKPOINT_DIR, type=click.Path(file_okay=False, dir_okay=True),
              help="Model checkpoint directory.")
@click.option("--checkpoint-period", default=CHECKPOINT_PERIOD, type=click.INT,
              help="Interval (number of epochs) between checkpoints.",)
@click.option("--summary-dir", default=SUMMARY_DIR, type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("-s", "--seed", default=SEED, type=click.INT, help="Random seed")
def main(name, initial_epoch, epochs, batch_size, l1_factor, l2_factor, 
         test_size, checkpoint_dir, checkpoint_period, summary_dir, seed):

    # Data loading and preprocessing
    X, y = load_zalando()
    # X = scale(X)
    X = normalize(X, norm='l1', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=seed)

    # Model specification

    model_filename_fmt = f"{name}.{{epoch:02d}}.h5"
    model_filename = model_filename_fmt.format(epoch=initial_epoch)

    if initial_epoch > 0 and os.path.isfile(os.path.join(checkpoint_dir, model_filename)):

        click.secho(f"Found model checkpoint [{model_filename}]. "
                    f"Resuming from epoch [{initial_epoch}]...", fg='green')

        model = load_model(os.path.join(checkpoint_dir, model_filename))

    else:
        click.secho(f"Could not load model checkpoint [{model_filename}]. "
                    "Building new model...", fg='green')

        model = build_model(output_dim=1, kernel_regularizer=l1_l2(l1_factor, l2_factor))
        model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

        initial_epoch = 0
        
    # Model fitting
    tensorboard_callback = TensorBoard(os.path.join(summary_dir, name), profile_batch=0)
    checkpoint_callback = ModelCheckpoint(os.path.join(checkpoint_dir, model_filename_fmt), period=checkpoint_period)
    csv_callback = CSVLogger(os.path.join(summary_dir, f"{name}.csv"), append=True)

    hist = model.fit(
        X_train, y_train, batch_size=batch_size, epochs=epochs,
        validation_data=(X_test, y_test), shuffle=True, initial_epoch=initial_epoch,
        callbacks=[tensorboard_callback, checkpoint_callback, csv_callback]
    )

    # for clf_name, clf in zip(classifier_names, classifiers):
    #     print(clf_name, clf)
    #     print(cross_val_score(clf, X_train, y_train, cv=2))
    #     print()

    # click.echo("Replace this message by putting your code into "
    #            "pearls.cli.main")
    # click.echo("See click documentation at https://click.palletsprojects.com/")
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
