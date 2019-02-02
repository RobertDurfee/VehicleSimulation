from zipfile import ZipFile
import argparse
from tempfile import TemporaryDirectory, mkdtemp
from glob import glob
import pandas as pd
from uuid import uuid4
import re
import numpy as np
from os import path, makedirs, listdir
import random
from google.cloud import storage
from shutil import rmtree
from ..utils import copy_to_gcs


def extract(zip_file, out_dir):
    """Takes a zipped file with all dynamometer data as provided from ANL and
    extracts it into the specified output directory.

    Args:
        zip_file (str): Path to zipped file.
        out_dir (str): Path to output directory for unzipped contents.

    """
    with ZipFile(zip_file, 'r') as zip_file:
        zip_file.extractall(out_dir)


def load(data_dir, in_features, out_features, missing_action='drop'):
    """Load all TSVs from ANL D3 extracted data file. Select only the
    necessary features.

    Args:
        data_dir (str): Directory containing ANL D3 extended data file (unzipped).
        in_features (list of str): Input features to extract from data set.
        out_features (list of str): Target features to extract from data set.
        missing_action (str): What to do with data sets with missing columns.
            - 'drop': Skip data files with missing columns (if few
                missing columns expected).
            - 'null': Fill the missing columns with NaN.

    Returns:
        DataFrame: All files concatenated containing selected features.

    """
    data_files = glob(path.join(data_dir, '**/*Test Data.txt'), recursive=True)

    dfs = []

    for data_file in data_files:

        df = pd.read_csv(data_file, sep='\t', header=0)

        # Identify missing columns
        def is_missing(feature): return feature not in df.columns
        missing_features = list(filter(is_missing, set(in_features + out_features)))

        # Handle missing columns
        if len(missing_features) > 0:
            if missing_action == 'drop':
                continue
            elif missing_action == 'null':
                for missing_feature in missing_features:
                    df[missing_feature] = np.NaN
            else:
                raise ValueError('\'' + missing_action + '\' is an unknown missing action.')

        # Select only the specified features
        df = df[in_features + out_features]

        # Create a multi-index
        multi_index_tuples = [('Input', in_feature) for in_feature in in_features] + \
                             [('Target', out_feature) for out_feature in out_features]
        df.columns = pd.MultiIndex.from_tuples(multi_index_tuples)

        # Shift targets
        df['Target'] = df['Target'].shift(-1)

        # Shift inputs which are not also used as outputs
        in_features_shift = filter(lambda in_feature: in_feature not in out_features, in_features)
        in_features_shift = [('Input', in_feature_shift) for in_feature_shift in in_features_shift]
        df[in_features_shift] = df[in_features_shift].shift(-1)

        # Remove the last value which will have NaN values
        df = df[:-1]

        # Set the index to be the TestID taken from file name (or guid if
        # unknown file name pattern)
        match = re.search(r'\d{8}', data_file)
        df['Test_ID[]'] = match.group(0) if match else str(uuid4())
        df.set_index('Test_ID[]', inplace=True)

        dfs.append(df)

    return pd.concat(dfs, sort=False)


def split(df, test_frac=0.1, shuffle=True):
    """Splits the data frame into train and test data frames based unique
    index values.

    Args:
        df (DataFrame): Data to split. Has `Test_ID[]` index.
        test_frac (float): Percent of data to use as test data.

    Returns:
        DataFrame: Training data.
        DataFrame: Testing data.

    """
    groups = list(df.groupby(df.index))
    split_ind = len(groups) - int(test_frac * len(groups))

    if shuffle:
        random.shuffle(groups)

    train_groups, test_groups = groups[:split_ind], groups[split_ind:]

    train_df = pd.concat([train_group[1] for train_group in train_groups], sort=False)
    test_df = pd.concat([test_group[1] for test_group in test_groups], sort=False)

    return train_df, test_df


def main(data_zip, job_dir, in_features, out_features, missing_action='drop',
         test_split=0.10, shuffle=True):
    """Preprocess ANL D3 data files. Extract zipped files, load requested
    features, split into train and test splits, and save all data.

    Args:
        data_zip (str): Location of ANL D3 zipped data.
        job_dir (str): Output directory for data files (local or remote GCS URI).
        in_features (list of str): Input features to extract from data set.
        out_features (list of str): Target features to extract from data set.
        missing_action (str): What to do with data sets with missing columns.
            - 'drop': Skip data files with missing columns (if few
                missing columns expected).
            - 'null': Fill the missing columns with NaN.
        test_split (float): Percent of data to withold for testing.
        shuffle (bool): Whether to shuffle the data before splitting or not.

    """
    with TemporaryDirectory() as temporary_directory:

        print('Uncompressing data...')
        extract(data_zip, temporary_directory)

        print('Extracting features...')
        df = load(temporary_directory, in_features, out_features, missing_action)

        # If the `job_dir` is on GCS, save everything to temporary folder and
        # upload at the end.
        if job_dir.startswith('gs://'):
            data_path = mkdtemp()
        else:
            data_path = path.join(job_dir, 'data')

        makedirs(data_path, exist_ok=True)

        print('Saving all data locally...')
        df.to_csv(path.join(data_path, 'all.csv'))

        print('Splitting into train/test sets...')
        train_df, test_df = split(df, test_split, shuffle)

        print('Saving training data locally...')
        train_df.to_csv(path.join(data_path, 'train.csv'))

        print('Saving test data locally...')
        test_df.to_csv(path.join(data_path, 'test.csv'))

        if job_dir.startswith('gs://'):

            print('Copying to GCS...')

            filenames = listdir(data_path)

            for i in range(len(filenames)):

                print('File {} of {}'.format(i + 1, len(filenames)))

                local_file = path.join(data_path, filenames[i])
                remote_file = path.join(job_dir, 'data', filenames[i])

                copy_to_gcs(local_file, remote_file)

            rmtree(data_path)
        
        print('Done')


if __name__ == '__main__':
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--data-zip',
        type=str,
        default='./Data/2017FordF150Ecoboost.zip',
        help='Location of ANL D3 extended data zip file.')
    parser.add_argument(
        '--job-dir',
        type=str,
        default='./Jobs/Sequence-2017FordF150Ecoboost',
        help='Directory to save extracted data.')
    parser.add_argument(
        '--in-features',
        nargs='+',
        type=str,
        default=[ 'Pedal_accel_pos_CAN[per]', 'Brake_pressure_applied_PCM[]' ],
        help='Input features to extract from the data set.')
    parser.add_argument(
        '--out-features',
        nargs='+',
        type=str,
        default=[ 'Dyno_Spd[mph]' ],
        help='Target features to extract from the data set.')
    parser.add_argument(
        '--missing-action',
        type=str,
        default='drop',
        help='What to do if column missing. Can be \'null\' or \'drop\'. \'null\' will fill column with \'NaN\'. \'drop\' will drop the data set with the missing column.')
    parser.add_argument(
        '--test-split',
        type=float,
        default=0.10,
        help='What percent of the data to using for testing.')
    parser.add_argument(
        '--shuffle',
        dest='shuffle',
        action='store_true',
        help='Indicates the data should be shuffled prior to splitting.')
    parser.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='Indicates the data should not be shuffled prior to splitting.')
    parser.set_defaults(shuffle=True)

    args, _ = parser.parse_known_args()
    main(**vars(args))
