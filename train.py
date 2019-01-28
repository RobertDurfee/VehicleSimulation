import pandas
import re
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import InputLayer, Masking, LSTM, TimeDistributed, Dense
from math import ceil
from keras.callbacks import Callback, ModelCheckpoint
import argparse
import json


def load_data(root_dir, ignores=[]):
    """Imports the dynamometer data ignoring specified files.
    
    Args:
        root_dir (str): Where the dynamometer data is located.
        ignore (:obj:`list` of :obj:`str`): File names to disregard.
        
    Returns:
        (:obj:`list` of :obj:`DataFrame`): Loaded dynamometer data.
        
    """    
    is_test_data = lambda file: re.match(r'^\d{8} Test Data\.txt$', file) is not None
    csvs = filter(is_test_data, os.listdir(root_dir))
    
    is_ignored = lambda csv: csv not in ignores
    csvs = filter(is_ignored, csvs)
    
    return [pandas.read_csv(root_dir + '/' + csv, sep='\t', header=0) for csv in csvs]


def extract_cols(dfs, x_cols, y_cols, pad_val):
    """Takes the data frames and extracts specified columns into X and Y 3D arrays.
    
    Note:
        The first measurement is removed to be used as an input. For example, if 
        your data has the following form
        ```
            a = [ 0, 1, 2, 3, 4 ]
        ```
        and you wish to use `a` as both an input and an output and the input was 
        marked to offset, it will be split like
        ```
            in  = [ 0, 1, 2, 3 ]
            out = [ 1, 2, 3, 4 ]
        ```
        This is also the case for columns only used as inputs. So if `a` was *only* 
        an input and not marked as an offset, it will be split like
        ```
            in = [ 1, 2, 3, 4 ]
        ```
        The same is the case for all outputs.
    
    Args:
        dfs (:obj:`list` of :obj:`DataFrame`): Data frames of complete dynamometer 
            data. Every column must be included.
        x_cols (:obj:`list` of (:obj:`str`, :obj:`bool`)): Pairs of column names 
            and if they should be offset. (e.g. an previous output is an input)
        y_cols (:obj:`list` of :obj:`str`): Pairs of column names.
        pad_val (float): Value for padding endings of short sequences. This should 
            be out of the valid range of data.
    
    Returns:
        (:obj:`ndarray`): Inputs in the form (n_samples, max_time_steps, len(x_cols)).
        (:obj:`ndarray`): Outputs in the form (n_samples, max_time_steps, len(y_cols)).
    
    """
    max_length = (ceil(max([len(df) for df in dfs]) / 100))
    
    X = np.full([len(dfs), max_length, len(x_cols)], pad_val)
    Y = np.full([len(dfs), max_length, len(y_cols)], pad_val)
    
    for i, df in enumerate(dfs):
        
        for k, (column_name, is_offset) in enumerate(x_cols):
            
            if is_offset:
                X[i,:len(df)-1,k] += df[column_name].values[:-1] - pad_val
            else:
                X[i,:len(df)-1,k] += df[column_name].values[1:] - pad_val

        for k, column_name in enumerate(y_cols):            
            Y[i,:len(df)-1,k] += df[column_name].values[1:] - pad_val
    
    return X, Y


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train LSTM RNN model for sequence to sequence learning of vehicle speed from brake and accelerator position using chasis dynamometer data.')
    
    parser.add_argument('--data-dir', default='./DynamometerData', help='')
    parser.add_argument('--ignores', default='[]', help='')
    parser.add_argument('--inputs', help='')
    parser.add_argument('--outputs', help='')
    parser.add_argument('--pad-val', default='-1.', help='')

    args = parser.parse_args()
    
    args.ignores = json.loads(args.ignores)
    args.inputs = json.loads(args.inputs)
    args.outputs = json.loads(args.outputs)
    args.pad_val = float(args.pad_val)
        
    dfs = load_data(args.data_dir, args.ignores)    
    X, Y = extract_cols(dfs, args.inputs, args.outputs, args.pad_val)