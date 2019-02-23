import re
import os
import pandas as pd
from math import ceil
import numpy as np
from os import path
from shutil import rmtree
from sequence.utils import copy_from_gcs
from tempfile import mkdtemp


def load_data(data_file, pad_val=-1.):
    """Loads data CSV into input and target ndarrays of shape (n_samples,
    max_timesteps, features).

    Args:
        data_file (str): Local or remote CSV file containing data to load.
        pad_val (float): Value to use for padding suffixes of sequences less
            than max_timesteps. This should be an invalid input.

    Returns:
        ndarray: Input data of shape (n_samples, max_timesteps, in_features).
        list of str: Input feature names.
        ndarray: Target data of shape (n_samples, max_timesteps, out_features).
        list of str: Target feature names.

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

    df = pd.read_csv(path.join(dirname, filename), header=[0, 1], index_col=0)

    groups = df.groupby(df.index)

    n_samples = len(groups)
    max_timesteps = max(len(group[1]) for group in groups)
    in_features = len(df['Input'].columns)
    out_features = len(df['Target'].columns)

    X = np.full([n_samples, max_timesteps, in_features], pad_val)
    Y = np.full([n_samples, max_timesteps, out_features], pad_val)

    for i, (_, group_df) in enumerate(groups):

        X[i, :len(group_df), :] += group_df['Input'].values - pad_val
        Y[i, :len(group_df), :] += group_df['Target'].values - pad_val

    # Remove temporary directory if remote
    if data_file.startswith('gs://'):
        rmtree(dirname)

    return X, list(df['Input'].columns), Y, list(df['Target'].columns)


def pad(X, batch_size, pad_val=-1.):
    """Pad the sequences such that batches evenly divide the data.

    Args:
        X (ndarray): Data to pad (n_samples, max_timesteps, features)
        batch_size (int): Length of sequences in batches desired.
        pad_val (float): Pad value.

    Returns:
        ndarry: Padded sequences now evenly divisible by batch_size.

    """
    n_samples, old_max_timesteps, features = X.shape

    complete_batches, left_over = divmod(old_max_timesteps, batch_size)

    if left_over > 0:
        new_max_timesteps = batch_size * (complete_batches + 1)
    else:
        new_max_timesteps = batch_size * complete_batches

    X_pad = np.full([n_samples, new_max_timesteps, features], pad_val)
    X_pad[:, :old_max_timesteps, :] = X

    return X_pad


def scale(X, min=0., max=1., pad_old=None, pad_new=None):
    """Scale sequence data into a specified range.

    Args:
        X (ndarray): Array of input of shape (n_samples, max_timesteps,
            features).
        min (float): New minimum for data.
        max (float): New maximum for data.
        pad_old (float): A value to consider as existing padding
            which should not be scaled.
        pad_new (float): Replace the old padding with this new value.
            Essentially a controlled padding.

    Return:
        ndarray: Scaled data in range [min, max].
        list of float: The original data minimums for each feature (ignoring padding).
        list of float: The original data maximums for each feature (ignoring padding).

    """
    n_samples, max_timesteps, features = X.shape

    mins = []
    maxs = []

    X_std = np.zeros([n_samples, max_timesteps, features])

    for k in range(features):

        # No values to ignore
        if pad_old is None:

            mins.append(X[:, :, k].min())
            maxs.append(X[:, :, k].max())
        
        # Ignore pad values
        else:

            mins.append(X[:, :, k][X[:, :, k] != pad_old].min())
            maxs.append(X[:, :, k][X[:, :, k] != pad_old].max())
        
        # Scale all values
        X_std[:, :, k] += ((X[:, :, k] - mins[k]) / (maxs[k] - mins[k])) * (max - min) + min

        # Reset padded values to the old or new pad value
        X_std[:, :, k][X[:, :, k] == pad_old] = pad_old if pad_new is None else pad_new

    return X_std, mins, maxs
