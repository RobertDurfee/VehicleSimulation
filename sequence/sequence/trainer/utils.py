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


def deduce_look_back(in_features, target_features):
    """From the feature names, determine how large of a look back is used.

    Args:
        in_features (list of str): Names of input features
        target_features (list of str): Names of target features.
    
    Returns:
        int: Number of look back features.
        int: Look back value.
    
    """
    def is_shared(target_feature):

        for in_feature in in_features:

            if re.match(re.escape(target_feature) + r'\d+$', in_feature):
                return True
        
        return False

    shared_features = list(filter(is_shared, target_features))

    if len(shared_features) == 0:
        return 0, None

    look_backs = []

    for shared_feature in shared_features:

        look_backs.append(0)

        for in_feature in in_features:

            if re.match(re.escape(shared_feature) + r'\d+$', in_feature):
                look_backs[-1] += 1
    
    if look_backs.count(look_backs[0]) != len(look_backs):
        raise ValueError('Inconsistent look back.')

    return len(look_backs), look_backs[0]


def update_prediction_history(Y_hist, Y_t):
    """Rolls Y_hist values toward the front and then replaces the last
    element with Y_t.
    
    Args:
        Y_hist (ndarray): Historical predictions with dimensions (n_samples,
            look_back_length, target_features) where the first element on
            axis 1 is the oldest look back and the last element is the
            newest.
        Y_t (ndarray): New prediction with the dimensions (n_samples, 1,
            target_features) to be added to history.
    
    Returns:
        ndarray: New historical values with dimensions (n_samples,
            look_back_length, target_features) with the newest prediction
            added to the end and the oldest is forgotten.
            
    """
    # Roll the historical predictions such that the oldest is now the last
    # element.
    Y_hist = np.roll(Y_hist, -1, axis=1)
    
    # Update the last element in historical predictions with the most recent.
    Y_hist[:, -1:, :] = Y_t

    return Y_hist


def replace_expected_look_back(X_t, Y_hist, n_look_back_features):
    """For the given X input, replace the expected look back values
    with the actual, observed values from prediction history.
    
    Args:
        X_t (ndarray): Current input with dimensions (n_samples, 1,
            in_features) to replace expected look back.
        Y_hist (ndarray): Past predictions with dimensions (n_samples,
            look_back_length, out_features) to use for look back.
        n_look_back_features (int): Number of features used for look
            back. These features are taken from the last elements
            of Y_hist.
    
    Returns:
        ndarray: New X_t with the appropriate expected look back values
            replaced with actual look backs.
            
    """
    look_back_length = Y_hist.shape[1]
    
    # Take only the look back features from previous predictions.
    look_back_vals = Y_hist[:, :, -n_look_back_features:]
    
    # Swap timesteps axis with features axis so the reshaping
    # works out correctly.
    look_back_vals = np.swapaxes(look_back_vals, 1, 2)
    
    # Flatten feature and timestep axes. The resulting shape
    # will be (n_samples, 1, features * timesteps). The format for
    # each sample is f1_t1, f1_t2, f1_t3, ..., f2_t1, f2_t2, 
    # f2_t2, ... and so on. The most recent output is the last
    # timestep for each feature.
    look_back_vals = look_back_vals.reshape(look_back_vals.shape[0], 1, -1)
    
    # Copy X_t to avoid sideffects
    X_t = X_t.copy()
    
    # Replace *expected* look back features with actual
    X_t[:, :, -(n_look_back_features * look_back_length):] = look_back_vals
    
    return X_t 
