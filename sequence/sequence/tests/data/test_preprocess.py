from unittest import TestCase
from tempfile import mkdtemp
import os
from os import path
from shutil import rmtree
from sequence.data.preprocess import load, split
import numpy as np
from math import isnan
import pandas as pd


class TestLoadSharedComplete(TestCase):

    def setUp(self):

        self.in_features = [ 'FeatureA', 'FeatureB', 'FeatureC' ]
        self.out_features = [ 'FeatureD', 'FeatureC' ]

        self.sequence_lengths = [ 2, 5, 4 ]

        self.feature_values = {
            ('Input', 'FeatureA'): [
                11.1, 12.1,
                21.1, 22.1, 23.1, 24.1, 25.1,
                31.1, 32.1, 33.1, 34.1
            ],
            ('Input', 'FeatureB'): [
                11.2, 12.2,
                21.2, 22.2, 23.2, 24.2, 25.2,
                31.2, 32.2, 33.2, 34.2
            ],
            ('Input', 'FeatureC'): [
                10.3, 11.3,
                20.3, 21.3, 22.3, 23.3, 24.3,
                30.3, 31.3, 32.3, 33.3
            ],
            ('Target', 'FeatureD'): [
                11.4, 12.4,
                21.4, 22.4, 23.4, 24.4, 25.4,
                31.4, 32.4, 33.4, 34.4
            ],
            ('Target', 'FeatureC'): [
                11.3, 12.3,
                21.3, 22.3, 23.3, 24.3, 25.3,
                31.3, 32.3, 33.3, 34.3
            ]
        }
        
        self.index_values = [
            '61700001', '61700001',
            '61700002', '61700002', '61700002', '61700002', '61700002',
            '61700003', '61700003', '61700003', '61700003'
        ]


        self.data_lines_dict = {
            '61700001': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '10.1\t10.2\t10.3\t10.4\t10.5\n',
                '11.1\t11.2\t11.3\t11.4\t11.5\n',
                '12.1\t12.2\t12.3\t12.4\t12.5\n'
            ],
            '61700002': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '20.1\t20.2\t20.3\t20.4\t20.5\n',
                '21.1\t21.2\t21.3\t21.4\t21.5\n',
                '22.1\t22.2\t22.3\t22.4\t22.5\n',
                '23.1\t23.2\t23.3\t23.4\t23.5\n',
                '24.1\t24.2\t24.3\t24.4\t24.5\n',
                '25.1\t25.2\t25.3\t25.4\t25.5\n'
            ],
            '61700003': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '30.1\t30.2\t30.3\t30.4\t30.5\n',
                '31.1\t31.2\t31.3\t31.4\t31.5\n',
                '32.1\t32.2\t32.3\t32.4\t32.5\n',
                '33.1\t33.2\t33.3\t33.4\t33.5\n',
                '34.1\t34.2\t34.3\t34.4\t34.5\n',
            ]
        }

        self.data_dir = mkdtemp()

        for test_id, data_lines in self.data_lines_dict.items():

            with open(path.join(self.data_dir, test_id + ' Test Data.txt'), 'w') as test_data_file:
                test_data_file.writelines(data_lines)

    def tearDown(self):
        rmtree(self.data_dir)
    
    def test_shape(self):

        df = load(self.data_dir, self.in_features, self.out_features)

        self.assertEqual(df.shape, (sum(self.sequence_lengths), sum([len(self.in_features), len(self.out_features)])))

    def test_columns(self):

        df = load(self.data_dir, self.in_features, self.out_features)

        self.assertTrue('Input' in df)
        self.assertTrue('Target' in df)

        self.assertEqual(df.index.name, 'Test_ID[]')
        self.assertEqual(list(df['Input'].columns), self.in_features)
        self.assertEqual(list(df['Target'].columns), self.out_features)

    def test_values(self):

        df = load(self.data_dir, self.in_features, self.out_features)

        self.assertEqual(list(df.index.values), self.index_values)
        
        for feature, feature_values in self.feature_values.items():
            self.assertEqual(list(df[feature].values), feature_values)


class TestLoadSharedIncompleteDrop(TestCase):

    def setUp(self):

        self.in_features = [ 'FeatureA', 'FeatureB', 'FeatureC' ]
        self.out_features = [ 'FeatureD', 'FeatureC' ]

        self.sequence_lengths = [ 2, 4 ]

        self.feature_values = {
            ('Input', 'FeatureA'): [
                11.1, 12.1,
                31.1, 32.1, 33.1, 34.1
            ],
            ('Input', 'FeatureB'): [
                11.2, 12.2,
                31.2, 32.2, 33.2, 34.2
            ],
            ('Input', 'FeatureC'): [
                10.3, 11.3,
                30.3, 31.3, 32.3, 33.3
            ],
            ('Target', 'FeatureD'): [
                11.4, 12.4,
                31.4, 32.4, 33.4, 34.4
            ],
            ('Target', 'FeatureC'): [
                11.3, 12.3,
                31.3, 32.3, 33.3, 34.3
            ]
        }
        
        self.index_values = [
            '61700001', '61700001',
            '61700003', '61700003', '61700003', '61700003'
        ]


        self.data_lines_dict = {
            '61700001': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '10.1\t10.2\t10.3\t10.4\t10.5\n',
                '11.1\t11.2\t11.3\t11.4\t11.5\n',
                '12.1\t12.2\t12.3\t12.4\t12.5\n'
            ],
            '61700002': [
                'FeatureA\tFeatureC\tFeatureD\tFeatureE\n',
                '20.1\t20.3\t20.4\t20.5\n',
                '21.1\t21.3\t21.4\t21.5\n',
                '22.1\t22.3\t22.4\t22.5\n',
                '23.1\t23.3\t23.4\t23.5\n',
                '24.1\t24.3\t24.4\t24.5\n',
                '25.1\t25.3\t25.4\t25.5\n'
            ],
            '61700003': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '30.1\t30.2\t30.3\t30.4\t30.5\n',
                '31.1\t31.2\t31.3\t31.4\t31.5\n',
                '32.1\t32.2\t32.3\t32.4\t32.5\n',
                '33.1\t33.2\t33.3\t33.4\t33.5\n',
                '34.1\t34.2\t34.3\t34.4\t34.5\n',
            ]
        }

        self.data_dir = mkdtemp()

        for test_id, data_lines in self.data_lines_dict.items():

            with open(path.join(self.data_dir, test_id + ' Test Data.txt'), 'w') as test_data_file:
                test_data_file.writelines(data_lines)

    def tearDown(self):
        rmtree(self.data_dir)
    
    def test_shape(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='drop')

        self.assertEqual(df.shape, (sum(self.sequence_lengths), sum([len(self.in_features), len(self.out_features)])))

    def test_columns(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='drop')

        self.assertTrue('Input' in df)
        self.assertTrue('Target' in df)

        self.assertEqual(df.index.name, 'Test_ID[]')
        self.assertEqual(list(df['Input'].columns), self.in_features)
        self.assertEqual(list(df['Target'].columns), self.out_features)

    def test_values(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='drop')

        self.assertEqual(list(df.index.values), self.index_values)
        
        for feature, feature_values in self.feature_values.items():
            self.assertEqual(list(df[feature].values), feature_values)


class TestLoadSharedIncompleteNull(TestCase):

    def setUp(self):

        self.in_features = [ 'FeatureA', 'FeatureB', 'FeatureC' ]
        self.out_features = [ 'FeatureD', 'FeatureC' ]

        self.sequence_lengths = [ 2, 5, 4 ]

        self.feature_values = {
            ('Input', 'FeatureA'): [
                11.1, 12.1,
                21.1, 22.1, 23.1, 24.1, 25.1,
                31.1, 32.1, 33.1, 34.1
            ],
            ('Input', 'FeatureB'): [
                11.2, 12.2,
                np.nan, np.nan, np.nan, np.nan, np.nan,
                31.2, 32.2, 33.2, 34.2
            ],
            ('Input', 'FeatureC'): [
                10.3, 11.3,
                20.3, 21.3, 22.3, 23.3, 24.3,
                30.3, 31.3, 32.3, 33.3
            ],
            ('Target', 'FeatureD'): [
                11.4, 12.4,
                21.4, 22.4, 23.4, 24.4, 25.4,
                31.4, 32.4, 33.4, 34.4
            ],
            ('Target', 'FeatureC'): [
                11.3, 12.3,
                21.3, 22.3, 23.3, 24.3, 25.3,
                31.3, 32.3, 33.3, 34.3
            ]
        }
        
        self.index_values = [
            '61700001', '61700001',
            '61700002', '61700002', '61700002', '61700002', '61700002',
            '61700003', '61700003', '61700003', '61700003'
        ]


        self.data_lines_dict = {
            '61700001': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '10.1\t10.2\t10.3\t10.4\t10.5\n',
                '11.1\t11.2\t11.3\t11.4\t11.5\n',
                '12.1\t12.2\t12.3\t12.4\t12.5\n'
            ],
            '61700002': [
                'FeatureA\tFeatureC\tFeatureD\tFeatureE\n',
                '20.1\t20.3\t20.4\t20.5\n',
                '21.1\t21.3\t21.4\t21.5\n',
                '22.1\t22.3\t22.4\t22.5\n',
                '23.1\t23.3\t23.4\t23.5\n',
                '24.1\t24.3\t24.4\t24.5\n',
                '25.1\t25.3\t25.4\t25.5\n'
            ],
            '61700003': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '30.1\t30.2\t30.3\t30.4\t30.5\n',
                '31.1\t31.2\t31.3\t31.4\t31.5\n',
                '32.1\t32.2\t32.3\t32.4\t32.5\n',
                '33.1\t33.2\t33.3\t33.4\t33.5\n',
                '34.1\t34.2\t34.3\t34.4\t34.5\n',
            ]
        }

        self.data_dir = mkdtemp()

        for test_id, data_lines in self.data_lines_dict.items():

            with open(path.join(self.data_dir, test_id + ' Test Data.txt'), 'w') as test_data_file:
                test_data_file.writelines(data_lines)

    def tearDown(self):
        rmtree(self.data_dir)
    
    def test_shape(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='null')

        self.assertEqual(df.shape, (sum(self.sequence_lengths), sum([len(self.in_features), len(self.out_features)])))

    def test_columns(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='null')

        self.assertTrue('Input' in df)
        self.assertTrue('Target' in df)

        self.assertEqual(df.index.name, 'Test_ID[]')
        self.assertEqual(list(df['Input'].columns), self.in_features)
        self.assertEqual(list(df['Target'].columns), self.out_features)

    def test_values(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='null')

        self.assertEqual(list(df.index.values), self.index_values)
        
        for feature, feature_values in self.feature_values.items():

            for actual, expected in zip(list(df[feature].values), feature_values):

                if isnan(actual):
                    self.assertTrue(isnan(expected))
                else:
                    self.assertEqual(actual, expected)


class TestLoadUniqueComplete(TestCase):

    def setUp(self):

        self.in_features = [ 'FeatureA', 'FeatureB' ]
        self.out_features = [ 'FeatureD' ]

        self.sequence_lengths = [ 2, 5, 4 ]

        self.feature_values = {
            ('Input', 'FeatureA'): [
                11.1, 12.1,
                21.1, 22.1, 23.1, 24.1, 25.1,
                31.1, 32.1, 33.1, 34.1
            ],
            ('Input', 'FeatureB'): [
                11.2, 12.2,
                21.2, 22.2, 23.2, 24.2, 25.2,
                31.2, 32.2, 33.2, 34.2
            ],
            ('Target', 'FeatureD'): [
                11.4, 12.4,
                21.4, 22.4, 23.4, 24.4, 25.4,
                31.4, 32.4, 33.4, 34.4
            ]
        }
        
        self.index_values = [
            '61700001', '61700001',
            '61700002', '61700002', '61700002', '61700002', '61700002',
            '61700003', '61700003', '61700003', '61700003'
        ]


        self.data_lines_dict = {
            '61700001': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '10.1\t10.2\t10.3\t10.4\t10.5\n',
                '11.1\t11.2\t11.3\t11.4\t11.5\n',
                '12.1\t12.2\t12.3\t12.4\t12.5\n'
            ],
            '61700002': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '20.1\t20.2\t20.3\t20.4\t20.5\n',
                '21.1\t21.2\t21.3\t21.4\t21.5\n',
                '22.1\t22.2\t22.3\t22.4\t22.5\n',
                '23.1\t23.2\t23.3\t23.4\t23.5\n',
                '24.1\t24.2\t24.3\t24.4\t24.5\n',
                '25.1\t25.2\t25.3\t25.4\t25.5\n'
            ],
            '61700003': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '30.1\t30.2\t30.3\t30.4\t30.5\n',
                '31.1\t31.2\t31.3\t31.4\t31.5\n',
                '32.1\t32.2\t32.3\t32.4\t32.5\n',
                '33.1\t33.2\t33.3\t33.4\t33.5\n',
                '34.1\t34.2\t34.3\t34.4\t34.5\n',
            ]
        }

        self.data_dir = mkdtemp()

        for test_id, data_lines in self.data_lines_dict.items():

            with open(path.join(self.data_dir, test_id + ' Test Data.txt'), 'w') as test_data_file:
                test_data_file.writelines(data_lines)

    def tearDown(self):
        rmtree(self.data_dir)
    
    def test_shape(self):

        df = load(self.data_dir, self.in_features, self.out_features)

        self.assertEqual(df.shape, (sum(self.sequence_lengths), sum([len(self.in_features), len(self.out_features)])))

    def test_columns(self):

        df = load(self.data_dir, self.in_features, self.out_features)

        self.assertTrue('Input' in df)
        self.assertTrue('Target' in df)

        self.assertEqual(df.index.name, 'Test_ID[]')
        self.assertEqual(list(df['Input'].columns), self.in_features)
        self.assertEqual(list(df['Target'].columns), self.out_features)

    def test_values(self):

        df = load(self.data_dir, self.in_features, self.out_features)

        self.assertEqual(list(df.index.values), self.index_values)
        
        for feature, feature_values in self.feature_values.items():
            self.assertEqual(list(df[feature].values), feature_values)


class TestLoadUniqueIncompleteDrop(TestCase):

    def setUp(self):

        self.in_features = [ 'FeatureA', 'FeatureB' ]
        self.out_features = [ 'FeatureD' ]

        self.sequence_lengths = [ 2, 4 ]

        self.feature_values = {
            ('Input', 'FeatureA'): [
                11.1, 12.1,
                31.1, 32.1, 33.1, 34.1
            ],
            ('Input', 'FeatureB'): [
                11.2, 12.2,
                31.2, 32.2, 33.2, 34.2
            ],
            ('Target', 'FeatureD'): [
                11.4, 12.4,
                31.4, 32.4, 33.4, 34.4
            ]
        }
        
        self.index_values = [
            '61700001', '61700001',
            '61700003', '61700003', '61700003', '61700003'
        ]


        self.data_lines_dict = {
            '61700001': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '10.1\t10.2\t10.3\t10.4\t10.5\n',
                '11.1\t11.2\t11.3\t11.4\t11.5\n',
                '12.1\t12.2\t12.3\t12.4\t12.5\n'
            ],
            '61700002': [
                'FeatureA\tFeatureC\tFeatureD\tFeatureE\n',
                '20.1\t20.3\t20.4\t20.5\n',
                '21.1\t21.3\t21.4\t21.5\n',
                '22.1\t22.3\t22.4\t22.5\n',
                '23.1\t23.3\t23.4\t23.5\n',
                '24.1\t24.3\t24.4\t24.5\n',
                '25.1\t25.3\t25.4\t25.5\n'
            ],
            '61700003': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '30.1\t30.2\t30.3\t30.4\t30.5\n',
                '31.1\t31.2\t31.3\t31.4\t31.5\n',
                '32.1\t32.2\t32.3\t32.4\t32.5\n',
                '33.1\t33.2\t33.3\t33.4\t33.5\n',
                '34.1\t34.2\t34.3\t34.4\t34.5\n',
            ]
        }

        self.data_dir = mkdtemp()

        for test_id, data_lines in self.data_lines_dict.items():

            with open(path.join(self.data_dir, test_id + ' Test Data.txt'), 'w') as test_data_file:
                test_data_file.writelines(data_lines)

    def tearDown(self):
        rmtree(self.data_dir)
    
    def test_shape(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='drop')

        self.assertEqual(df.shape, (sum(self.sequence_lengths), sum([len(self.in_features), len(self.out_features)])))

    def test_columns(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='drop')

        self.assertTrue('Input' in df)
        self.assertTrue('Target' in df)

        self.assertEqual(df.index.name, 'Test_ID[]')
        self.assertEqual(list(df['Input'].columns), self.in_features)
        self.assertEqual(list(df['Target'].columns), self.out_features)

    def test_values(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='drop')

        self.assertEqual(list(df.index.values), self.index_values)
        
        for feature, feature_values in self.feature_values.items():
            self.assertEqual(list(df[feature].values), feature_values)


class TestLoadUniqueIncompleteNull(TestCase):

    def setUp(self):

        self.in_features = [ 'FeatureA', 'FeatureB' ]
        self.out_features = [ 'FeatureD' ]

        self.sequence_lengths = [ 2, 5, 4 ]

        self.feature_values = {
            ('Input', 'FeatureA'): [
                11.1, 12.1,
                21.1, 22.1, 23.1, 24.1, 25.1,
                31.1, 32.1, 33.1, 34.1
            ],
            ('Input', 'FeatureB'): [
                11.2, 12.2,
                np.nan, np.nan, np.nan, np.nan, np.nan,
                31.2, 32.2, 33.2, 34.2
            ],
            ('Target', 'FeatureD'): [
                11.4, 12.4,
                21.4, 22.4, 23.4, 24.4, 25.4,
                31.4, 32.4, 33.4, 34.4
            ]
        }
        
        self.index_values = [
            '61700001', '61700001',
            '61700002', '61700002', '61700002', '61700002', '61700002',
            '61700003', '61700003', '61700003', '61700003'
        ]


        self.data_lines_dict = {
            '61700001': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '10.1\t10.2\t10.3\t10.4\t10.5\n',
                '11.1\t11.2\t11.3\t11.4\t11.5\n',
                '12.1\t12.2\t12.3\t12.4\t12.5\n'
            ],
            '61700002': [
                'FeatureA\tFeatureC\tFeatureD\tFeatureE\n',
                '20.1\t20.3\t20.4\t20.5\n',
                '21.1\t21.3\t21.4\t21.5\n',
                '22.1\t22.3\t22.4\t22.5\n',
                '23.1\t23.3\t23.4\t23.5\n',
                '24.1\t24.3\t24.4\t24.5\n',
                '25.1\t25.3\t25.4\t25.5\n'
            ],
            '61700003': [
                'FeatureA\tFeatureB\tFeatureC\tFeatureD\tFeatureE\n',
                '30.1\t30.2\t30.3\t30.4\t30.5\n',
                '31.1\t31.2\t31.3\t31.4\t31.5\n',
                '32.1\t32.2\t32.3\t32.4\t32.5\n',
                '33.1\t33.2\t33.3\t33.4\t33.5\n',
                '34.1\t34.2\t34.3\t34.4\t34.5\n',
            ]
        }

        self.data_dir = mkdtemp()

        for test_id, data_lines in self.data_lines_dict.items():

            with open(path.join(self.data_dir, test_id + ' Test Data.txt'), 'w') as test_data_file:
                test_data_file.writelines(data_lines)

    def tearDown(self):
        rmtree(self.data_dir)
    
    def test_shape(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='null')

        self.assertEqual(df.shape, (sum(self.sequence_lengths), sum([len(self.in_features), len(self.out_features)])))

    def test_columns(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='null')

        self.assertTrue('Input' in df)
        self.assertTrue('Target' in df)

        self.assertEqual(df.index.name, 'Test_ID[]')
        self.assertEqual(list(df['Input'].columns), self.in_features)
        self.assertEqual(list(df['Target'].columns), self.out_features)

    def test_values(self):

        df = load(self.data_dir, self.in_features, self.out_features, missing_action='null')

        self.assertEqual(list(df.index.values), self.index_values)
        
        for feature, feature_values in self.feature_values.items():

            for actual, expected in zip(list(df[feature].values), feature_values):

                if isnan(actual):
                    self.assertTrue(isnan(expected))
                else:
                    self.assertEqual(actual, expected)


class TestSplit(TestCase):

    def setUp(self):

        self.sequence_lengths = [ 2, 5, 4 ]

        self.in_features = [ 'FeatureA', 'FeatureB', 'FeatureC' ]
        self.out_features = [ 'FeatureD', 'FeatureC' ]

        self.feature_values = {
            ('Input', 'FeatureA'): [
                11.1, 12.1,
                21.1, 22.1, 23.1, 24.1, 25.1,
                31.1, 32.1, 33.1, 34.1
            ],
            ('Input', 'FeatureB'): [
                11.2, 12.2,
                21.2, 22.2, 23.2, 24.2, 25.2,
                31.2, 32.2, 33.2, 34.2
            ],
            ('Input', 'FeatureC'): [
                10.3, 11.3,
                20.3, 21.3, 22.3, 23.3, 24.3,
                30.3, 31.3, 32.3, 33.3
            ],
            ('Target', 'FeatureD'): [
                11.4, 12.4,
                21.4, 22.4, 23.4, 24.4, 25.4,
                31.4, 32.4, 33.4, 34.4
            ],
            ('Target', 'FeatureC'): [
                11.3, 12.3,
                21.3, 22.3, 23.3, 24.3, 25.3,
                31.3, 32.3, 33.3, 34.3
            ]
        }

        self.index_values = [
            '61700001', '61700001',
            '61700002', '61700002', '61700002', '61700002', '61700002',
            '61700003', '61700003', '61700003', '61700003'
        ]

        self.df = pd.DataFrame.from_dict(self.feature_values)
        self.df['Test_ID[]'] = self.index_values
        self.df.set_index('Test_ID[]', inplace=True)

    def test_shape(self):

        train_df, test_df = split(self.df, test_frac=1/3, shuffle=False)

        self.assertEqual(train_df.shape, (sum(self.sequence_lengths[:2]), len(self.feature_values)))
        self.assertEqual(test_df.shape, (sum(self.sequence_lengths[2:]), len(self.feature_values)))
    
    def test_columns(self):

        train_df, test_df = split(self.df, test_frac=1/3, shuffle=False)

        self.assertTrue('Input' in train_df)
        self.assertTrue('Target' in train_df)
        self.assertTrue('Input' in test_df)
        self.assertTrue('Target' in test_df)

        self.assertEqual(train_df.index.name, 'Test_ID[]')
        self.assertEqual(test_df.index.name, 'Test_ID[]')

        self.assertEqual(list(train_df['Input'].columns), self.in_features)
        self.assertEqual(list(train_df['Target'].columns), self.out_features)
        self.assertEqual(list(test_df['Input'].columns), self.in_features)
        self.assertEqual(list(test_df['Target'].columns), self.out_features)

    def test_values(self):

        train_df, test_df = split(self.df, test_frac=1/3, shuffle=False)

        self.assertEqual(list(train_df.index.values), self.index_values[:sum(self.sequence_lengths[:2])])
        self.assertEqual(list(test_df.index.values), self.index_values[sum(self.sequence_lengths[:2]):])

        for feature, feature_values in self.feature_values.items():

            self.assertEqual(list(train_df[feature].values), feature_values[:sum(self.sequence_lengths[:2])])
            self.assertEqual(list(test_df[feature].values), feature_values[sum(self.sequence_lengths[:2]):])
