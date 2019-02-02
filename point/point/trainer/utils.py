import re
import os
import pandas as pd
from math import ceil
import numpy as np
from os import path
from shutil import rmtree
from point.utils import copy_from_gcs
from tempfile import mkdtemp


def load_data(data_file):
    """Loads data CSV into input and target ndarrays of shape (n_samples,
    features).

    Args:
        data_file (str): Local or remote CSV file containing data to load.

    Returns:
        ndarray: Input data of shape (n_samples, in_features).
        ndarray: Target data of shape (n_samples, out_features).

    """
    # Copy remote file from GCS
    filename = path.basename(data_file)
    if data_file.startswith('gs://'):

        dirname = mkdtemp()
        local_file = path.join(dirname, filename)
        remote_file = data_file

        copy_from_gcs(remote_file, local_file)

    else:
        dirname = path.dirname(data_file)

    df = pd.read_csv(path.join(dirname, filename), header=[0, 1])

    X = df['Input'].values
    Y = df['Target'].values

    # Remove temporary directory if remote
    if data_file.startswith('gs://'):
        rmtree(dirname)

    return X, Y


def scale(X, min=0., max=1.):
    """Scale sequence data into a specified range.

    Args:
        X (ndarray): Array of input of shape (n_samples, max_timesteps,
            features).
        min (float): New minimum for data.
        max (float): New maximum for data.

    Return:
        ndarray: Scaled data in range [min, max].
        list of float: The original data minimums for each feature.
        list of float: The original data maximums for each feature.

    """
    n_samples, features = X.shape

    mins = []
    maxs = []

    X_std = np.zeros([n_samples, features])

    for k in range(features):

        mins.append(X[:, k].min())
        maxs.append(X[:, k].max())

        X_std[:, k] += ((X[:, k] - mins[k]) / (maxs[k] - mins[k])) * (max - min) + min

    return X_std, mins, maxs
