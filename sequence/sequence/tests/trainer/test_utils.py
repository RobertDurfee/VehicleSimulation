from unittest import TestCase
from sequence.trainer.utils import pad, scale, load_data, deduce_look_back, update_prediction_history
from tempfile import mkstemp
import numpy as np
import os


class TestPadDivisible(TestCase):

    def setUp(self):

        self.n_samples = 1
        self.max_timesteps = 100
        self.features = 1
        self.batch_size = 10

        self.X = np.random.randn(self.n_samples, self.max_timesteps, self.features)

    def test_no_side_effects(self):

        X_copy = self.X.copy()
        _ = pad(self.X, self.batch_size)

        self.assertTrue(np.allclose(self.X, X_copy))

    def test_shape(self):

        X_pad = pad(self.X, self.batch_size)

        self.assertEqual(X_pad.shape, (self.n_samples, self.max_timesteps, self.features))

    def test_values(self):

        X_pad = pad(self.X, self.batch_size)

        self.assertTrue(np.allclose(self.X, X_pad))


class TestPadNotDivisible(TestCase):

    def setUp(self):

        self.n_samples = 1
        self.max_timesteps = 95
        self.features = 1
        self.batch_size = 10

        self.X = np.random.randn(self.n_samples, self.max_timesteps, self.features)

    def test_no_side_effects(self):

        X_copy = self.X.copy()
        _ = pad(self.X, self.batch_size)

        self.assertTrue(np.allclose(self.X, X_copy))

    def test_shape(self):

        X_pad = pad(self.X, self.batch_size)

        self.assertEqual(X_pad.shape, (self.n_samples, 100, self.features))

    def test_values(self):

        X_pad = pad(self.X, self.batch_size)

        self.assertTrue(np.allclose(self.X, X_pad[:, :self.max_timesteps, :]))

    def test_default_pad_val(self):

        padding = np.full((self.n_samples, 5, self.features), -1.)

        X_pad = pad(self.X, self.batch_size)

        self.assertTrue(np.allclose(X_pad[:, self.max_timesteps:, :], padding))

    def test_custom_pad_val(self):

        pad_val = 20.
        padding = np.full((self.n_samples, 5, self.features), pad_val)

        X_pad = pad(self.X, self.batch_size, pad_val)

        self.assertTrue(np.allclose(X_pad[:, self.max_timesteps:, :], padding))


class TestScale(TestCase):

    def setUp(self):

        self.n_samples = 1
        self.max_timesteps = 5
        self.features = 2

        self.X = np.array([[[-6., -2.],
                            [ 3.,  5.],
                            [ 0.,  2.],
                            [-4., -1.],
                            [ 2.,  4.]]])

        self.feature_mins = [-6., -2.]
        self.feature_maxs = [ 3.,  5.]

    def test_no_side_effects(self):

        X_copy = self.X.copy()
        _ = scale(self.X)

        self.assertTrue(np.allclose(self.X, X_copy))

    def test_shape(self):

        X_scale, _, _ = scale(self.X)

        self.assertEqual(X_scale.shape, (self.n_samples, self.max_timesteps, self.features))

    def test_default_min(self):

        X_scale, _, _ = scale(self.X)

        self.assertEqual(X_scale.min(), 0.)

    def test_default_max(self):

        X_scale, _, _ = scale(self.X)

        self.assertEqual(X_scale.max(), 1.)

    def test_custom_min(self):

        min = -1.
        X_scale, _, _ = scale(self.X, min=min)

        self.assertEqual(X_scale.min(), min)

    def test_custom_max(self):

        max = 5.
        X_scale, _, _ = scale(self.X, max=max)

        self.assertEqual(X_scale.max(), max)

    def test_feature_mins(self):

        _, feature_mins, _ = scale(self.X)

        self.assertEqual(feature_mins, self.feature_mins)

    def test_feature_maxs(self):

        _, _, feature_maxs = scale(self.X)

        self.assertEqual(feature_maxs, self.feature_maxs)


class TestScalePadding(TestCase):

    def setUp(self):

        self.n_samples = 1
        self.max_timesteps = 4
        self.max_timesteps_padded = 7
        self.features = 2
        self.pad_old = -1.
        self.pad_new = -100.

        self.X = np.array([[[ 6.,  8.],
                            [ 3.,  5.],
                            [ 0.,  2.],
                            [ 4.,  0.],
                            [-1., -1.],
                            [-1., -1.],
                            [-1., -1.]]])

        self.feature_mins = [ 0.,  0.]
        self.feature_maxs = [ 6.,  8.]

    def test_no_side_effects(self):

        X_copy = self.X.copy()
        _ = scale(self.X, pad_old=self.pad_old)

        self.assertTrue(np.allclose(self.X, X_copy))

    def test_shape(self):

        X_scale, _, _ = scale(self.X, pad_old=self.pad_old)

        self.assertEqual(X_scale.shape, (self.n_samples, self.max_timesteps_padded, self.features))

    def test_default_min(self):

        X_scale, _, _ = scale(self.X, pad_old=self.pad_old)

        self.assertEqual(X_scale[:, :self.max_timesteps, :].min(), 0.)

    def test_default_max(self):

        X_scale, _, _ = scale(self.X, pad_old=self.pad_old)

        self.assertEqual(X_scale[:, :self.max_timesteps, :].max(), 1.)

    def test_custom_min(self):

        min = -1.
        X_scale, _, _ = scale(self.X, min=min, pad_old=self.pad_old)

        self.assertEqual(X_scale[:, :self.max_timesteps, :].min(), min)

    def test_custom_max(self):

        max = 5.
        X_scale, _, _ = scale(self.X, max=max, pad_old=self.pad_old)

        self.assertEqual(X_scale[:, :self.max_timesteps, :].max(), max)

    def test_feature_mins(self):

        _, feature_mins, _ = scale(self.X, pad_old=self.pad_old)

        self.assertEqual(feature_mins, self.feature_mins)

    def test_feature_maxs(self):

        _, _, feature_maxs = scale(self.X, pad_old=self.pad_old)

        self.assertEqual(feature_maxs, self.feature_maxs)
    
    def test_old_pad_values(self):

        X_scale, _, _ = scale(self.X, pad_old=self.pad_old)

        self.assertTrue(np.allclose(X_scale[:, self.max_timesteps:, :], self.X[:, self.max_timesteps:, :]))

    def test_new_pad_values(self):

        X_scale, _, _ = scale(self.X, pad_old=self.pad_old, pad_new=self.pad_new)

        self.assertTrue(np.allclose(X_scale[:, self.max_timesteps:, :], np.full((self.X.shape[0], self.max_timesteps_padded - self.max_timesteps, self.X.shape[2]), self.pad_new)))


class TestLoadData(TestCase):

    def setUp(self):

        self.n_samples = 3
        self.max_timesteps = 5
        self.in_features = 3
        self.out_features = 2

        self.in_seqs = [
            np.array([[11.1, 11.2, 10.3],
                      [12.1, 12.2, 11.3]]),

            np.array([[21.1, 21.2, 20.3],
                      [22.1, 22.2, 21.3],
                      [23.1, 23.2, 22.3],
                      [24.1, 24.2, 23.3],
                      [25.1, 25.2, 24.3]]),

            np.array([[31.1, 31.2, 30.3],
                      [32.1, 32.2, 31.3],
                      [33.1, 33.2, 32.3],
                      [34.1, 34.2, 33.3]])
        ]

        self.out_seqs = [
            np.array([[11.4, 11.3],
                      [12.4, 12.3]]),

            np.array([[21.4, 21.3],
                      [22.4, 22.3],
                      [23.4, 23.3],
                      [24.4, 24.3],
                      [25.4, 25.3]]),

            np.array([[31.4, 31.3],
                      [32.4, 32.3],
                      [33.4, 33.3],
                      [34.4, 34.3]])
        ]

        data_lines = [
            ',Input,Input,Input,Target,Target\n',
            ',FeatureA,FeatureB,FeatureC,FeatureD,FeatureC\n',
            'Ident,,,,,\n',

            '61700001,11.1,11.2,10.3,11.4,11.3\n',
            '61700001,12.1,12.2,11.3,12.4,12.3\n',

            '61700002,21.1,21.2,20.3,21.4,21.3\n',
            '61700002,22.1,22.2,21.3,22.4,22.3\n',
            '61700002,23.1,23.2,22.3,23.4,23.3\n',
            '61700002,24.1,24.2,23.3,24.4,24.3\n',
            '61700002,25.1,25.2,24.3,25.4,25.3\n'

            '61700003,31.1,31.2,30.3,31.4,31.3\n',
            '61700003,32.1,32.2,31.3,32.4,32.3\n',
            '61700003,33.1,33.2,32.3,33.4,33.3\n',
            '61700003,34.1,34.2,33.3,34.4,34.3\n',
        ]

        self.input_feature_names = [ 'FeatureA', 'FeatureB', 'FeatureC' ]
        self.target_feature_names = [ 'FeatureD', 'FeatureC' ]

        data_file, self.data_file_path = mkstemp()

        os.write(data_file, bytes(''.join(data_lines), 'ascii'))
        os.close(data_file)
    
    def tearDown(self):

        os.remove(self.data_file_path)
    
    def test_shape(self):

        X, _, Y, _ = load_data(self.data_file_path)

        self.assertEqual(X.shape, (self.n_samples, self.max_timesteps, self.in_features))
        self.assertEqual(Y.shape, (self.n_samples, self.max_timesteps, self.out_features))

    def test_values(self):

        X, _, Y, _ = load_data(self.data_file_path)

        for i in range(self.n_samples):

            timesteps, _ = self.in_seqs[i].shape
            self.assertTrue(np.allclose(X[i, :timesteps, :], self.in_seqs[i]))

            timesteps, _ = self.out_seqs[i].shape
            self.assertTrue(np.allclose(Y[i, :timesteps, :], self.out_seqs[i]))
    
    def test_default_pad_val(self):

        X, _, Y, _ = load_data(self.data_file_path)

        for i in range(self.n_samples):

            timesteps, _ = self.in_seqs[i].shape
            padding = np.full((self.max_timesteps - timesteps, self.in_features), -1.)

            self.assertTrue(np.allclose(X[i, timesteps:, :], padding))

            timesteps, _ = self.out_seqs[i].shape
            padding = np.full((self.max_timesteps - timesteps, self.out_features), -1.)

            self.assertTrue(np.allclose(Y[i, timesteps:, :], padding))
    
    def test_custom_pad_val(self):

        pad_val = -100.

        X, _, Y, _ = load_data(self.data_file_path, pad_val)

        for i in range(self.n_samples):

            timesteps, _ = self.in_seqs[i].shape
            padding = np.full((self.max_timesteps - timesteps, self.in_features), pad_val)

            self.assertTrue(np.allclose(X[i, timesteps:, :], padding))

            timesteps, _ = self.out_seqs[i].shape
            padding = np.full((self.max_timesteps - timesteps, self.out_features), pad_val)

            self.assertTrue(np.allclose(Y[i, timesteps:, :], padding))
    
    def test_feature_names(self):

        _, input_features, _, target_features = load_data(self.data_file_path)

        self.assertEqual(input_features, self.input_feature_names)
        self.assertEqual(target_features, self.target_feature_names)


class TestDeduceLookBack(TestCase):

    def test_is_none(self):

        in_features = [ 'A', 'B', 'C' ]
        target_features = [ 'D', 'E', 'F' ]

        self.assertEqual(deduce_look_back(in_features, target_features), (0, None))

    def test_is_correct_value(self):

        num_look_back_features = 2
        look_back = 4
        in_features = [ 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3', 'B4' ]
        target_features = [ 'A', 'B' ]

        self.assertEqual(deduce_look_back(in_features, target_features), (num_look_back_features, look_back))
    
    def test_raises_value_error(self):

        in_features = [ 'A1', 'A2', 'A3', 'A4', 'B1', 'B2', 'B3' ]
        target_features = [ 'A', 'B' ]

        self.assertRaises(ValueError, lambda: deduce_look_back(in_features, target_features))


class TestUpdatePredictionHistory(TestCase):

    def setUp(self):

        self.Y_hist = np.array([[[111, 112],
                                 [121, 122],
                                 [131, 132]],

                                [[211, 212],
                                 [221, 222],
                                 [231, 232]],

                                [[311, 312],
                                 [321, 322],
                                 [331, 332]],

                                [[411, 412],
                                 [421, 422],
                                 [431, 432]]])

        self.Y_t = np.array([[[141, 142]],

                             [[241, 242]],

                             [[341, 342]],

                             [[441, 442]]])

        self.Y_hist_updated = np.array([[[121, 122],
                                         [131, 132],
                                         [141, 142]],
          
                                        [[221, 222],
                                         [231, 232],
                                         [241, 242]],
          
                                        [[321, 322],
                                         [331, 332],
                                         [341, 342]],
          
                                        [[421, 422],
                                         [431, 432],
                                         [441, 442]]])
    
    def test_no_side_effects(self):

        Y_hist_copy = self.Y_hist.copy()
        Y_t_copy = self.Y_t.copy()

        _ = update_prediction_history(self.Y_hist, self.Y_t)

        self.assertTrue(np.allclose(self.Y_hist, Y_hist_copy))
        self.assertTrue(np.allclose(self.Y_t, Y_t_copy))

    def test_no_reference(self):

        Y_hist = update_prediction_history(self.Y_hist, self.Y_t)

        Y_hist[0, -1, 0] = -10000.

        self.assertNotEqual(Y_hist[0, -1, 0], self.Y_t[0, 0, 0])

    def test_shape(self):
    
        Y_hist = update_prediction_history(self.Y_hist, self.Y_t)

        self.assertEqual(Y_hist.shape, self.Y_hist.shape)

    def test_values(self):

        Y_hist = update_prediction_history(self.Y_hist, self.Y_t)

        self.assertTrue(np.allclose(Y_hist, self.Y_hist_updated))
