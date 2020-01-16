"""Console script for pearls."""
import sys
import os.path

import click

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, TensorBoard

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold

from pearls.datasets import load_zalando

# from sklearn.naive_bayes import GaussianNB
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neural_network import MLPClassifier
# from sklearn.svm import SVC
# from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
#                               GradientBoostingClassifier)

# Sensible defaults
EPOCHS = 50
BATCH_SIZE = 64

OPTIMIZER = "rmsprop"

BATCH_NORM = False
L1_FACTOR = 0.0
L2_FACTOR = 1e-5

SPLIT_METHOD = "shuffle"
N_SPLITS = 1
RESUME_FROM_EPOCH = 0

TEST_SIZE = 0.4
SEED = None

CHECKPOINT_DIR = "checkpoints/"
CHECKPOINT_PERIOD = 10

SUMMARY_DIR = "logs/"

# classifier_names = [
#     # "Naive Bayes",
#     # "Nearest Neighbors",
#     # "Logistic Regression",
#     "Neural Net",
#     # "Linear SVM",
#     # "RBF SVM",
#     # "Random Forest",
#     # "AdaBoost"
# ]

# classifiers = [
#     # GaussianNB(),
#     # KNeighborsClassifier(),
#     # LogisticRegression(),
#     MLPClassifier(hidden_layer_sizes=(64, 64), max_iter=200, verbose=True),
#     # SVC(kernel="linear", C=0.025),
#     # SVC(gamma=2, C=1),
#     # RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
#     # AdaBoostClassifier(),
# ]


def maybe_load_model(checkpoint_dir, model_filename_fmt, resume_from_epoch,
                     batch_norm, l1_factor, l2_factor, optimizer):
    """
    Attempt to load the specified model (including architecture, weights, and
    even optimizer states). If this is not possible, build a new model from
    scratch.
    """
    model_filename = model_filename_fmt.format(epoch=resume_from_epoch)
    checkpoint_path = os.path.join(checkpoint_dir, model_filename)

    if resume_from_epoch > 0 and os.path.isfile(checkpoint_path):

        click.secho(f"Found model checkpoint '{checkpoint_path}'. "
                    f"Resuming from epoch {resume_from_epoch}.", fg='green')

        model = load_model(checkpoint_path)

        initial_epoch = resume_from_epoch

    else:

        click.secho(f"Could not load model checkpoint '{checkpoint_path}' "
                    "or `resume_from_epoch == 0`. Building new model.",
                    fg='yellow')

        model = build_model(output_dim=1, batch_norm=batch_norm,
                            kernel_regularizer=l1_l2(l1_factor, l2_factor))
        model.compile(loss='binary_crossentropy', optimizer=optimizer,
                      metrics=['accuracy'])

        initial_epoch = 0

    return model, initial_epoch


def build_model(output_dim, num_layers=2, num_units=64, activation="relu",
                batch_norm=False, *args, **kwargs):
    """
    Instantiate a multi-layer perceptron with rectangular-shaped hidden
    layers.
    """
    model = Sequential()

    for l in range(num_layers):

        model.add(Dense(num_units, activation=activation, *args, **kwargs))

        if batch_norm:
            model.add(BatchNormalization())

    model.add(Dense(output_dim, activation="sigmoid"))

    return model


def build_callbacks(summary_dir, checkpoint_dir, basename, model_filename_fmt,
                    checkpoint_period):

    tensorboard_path = os.path.join(summary_dir, basename)
    csv_path = os.path.join(summary_dir, f"{basename}.csv")
    checkpoint_path = os.path.join(checkpoint_dir, model_filename_fmt)

    callbacks = []
    callbacks.append(TensorBoard(tensorboard_path, profile_batch=0))
    callbacks.append(CSVLogger(csv_path, append=True))
    callbacks.append(ModelCheckpoint(checkpoint_path, period=checkpoint_period))

    return callbacks


@click.command()
@click.argument("name")
@click.option("--optimizer", default=OPTIMIZER)
@click.option("-e", "--epochs", default=EPOCHS, type=int,
              help="Number of epochs.")
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=int,
              help="Batch size.")
@click.option('--skip-fit', is_flag=True,
              help="Skip model fitting. Only evaluate model.")
@click.option("--resume-from-epoch", default=RESUME_FROM_EPOCH, type=int,
              help="Epoch at which to resume a previous training run")
@click.option("--l1-factor", default=L1_FACTOR, type=float,
              help="L1 regularization factor.")
@click.option("--l2-factor", default=L2_FACTOR, type=float,
              help="L2 regularization factor.")
@click.option("--batch-norm/--no-batch-norm", default=BATCH_NORM,
              help="Use Batch Normalization (after activation).")
@click.option('--standardize', is_flag=True)
@click.option("--split-method", default=SPLIT_METHOD,
              type=click.Choice(["kfold", "shuffle"]),
              help="Method for generating train/test dataset splits.")
@click.option("--n-splits", default=N_SPLITS, type=int,
              help="Number of train/test dataset splits.")
@click.option("--test-size", default=TEST_SIZE, type=float,
              help="Test set size (for shuffle split method only).")
@click.option("--checkpoint-dir", default=CHECKPOINT_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Model checkpoint directory.")
@click.option("--checkpoint-period", default=CHECKPOINT_PERIOD, type=int,
              help="Interval (number of epochs) between checkpoints.")
@click.option("--summary-dir", default=SUMMARY_DIR,
              type=click.Path(file_okay=False, dir_okay=True),
              help="Summary directory.")
@click.option("-s", "--seed", default=SEED, type=int, help="Random seed")
def main(name, optimizer, epochs, batch_size, skip_fit, resume_from_epoch,
         l1_factor, l2_factor, batch_norm, standardize, split_method, n_splits,
         test_size, checkpoint_dir, checkpoint_period, summary_dir, seed):

    # Data loading
    X, y = load_zalando()

    if split_method == "kfold":
        splitter = StratifiedKFold(n_splits=n_splits,
                                   shuffle=True,
                                   random_state=seed)
    else:
        splitter = StratifiedShuffleSplit(n_splits=n_splits,
                                          test_size=test_size,
                                          random_state=seed)

    for split_num, (train_ind, test_ind) in enumerate(splitter.split(X, y)):

        X_train, y_train = X[train_ind], y[train_ind]
        X_test, y_test = X[test_ind], y[test_ind]

        # Data pre-processing
        if standardize:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # Model specification
        basename = f"{name}.split{split_num:d}"

        model_filename_fmt = f"{basename}.{{epoch:02d}}.h5"
        model, initial_epoch = maybe_load_model(checkpoint_dir,
                                                model_filename_fmt,
                                                resume_from_epoch, batch_norm,
                                                l1_factor, l2_factor,
                                                optimizer)

        # Model fitting
        if not skip_fit:

            callbacks = build_callbacks(summary_dir, checkpoint_dir, basename,
                                        model_filename_fmt, checkpoint_period)

            hist = model.fit(X_train, y_train, batch_size=batch_size,
                             epochs=epochs, validation_data=(X_test, y_test),
                             shuffle=True, initial_epoch=initial_epoch,
                             callbacks=callbacks)

        # Model evaluation
        loss_test, acc_test = model.evaluate(X_test, y_test)

        click.secho(f"[Split {split_num:d}] test accuracy: {acc_test:.3f}, "
                    f"test loss {loss_test:.3f}", fg='green')

    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
