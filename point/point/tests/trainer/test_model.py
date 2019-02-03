from unittest import TestCase
import numpy as np
from point.trainer.model import batch_generator, create_model, compile_model
import sys
from keras.models import Sequential
from keras.layers import InputLayer, Dense, LSTM, TimeDistributed
from keras.optimizers import Adam


class TestBatchGeneratorDivisible(TestCase):

    def setUp(self):

        self.n_samples = 6
        self.in_features = 3
        self.out_features = 2

        self.X = np.array([[-3.,  0., -8.],
                           [ 0., -2.,  3.],
                           [ 3., -3., -1.],
                           [-1.,  6.,  1.],
                           [-7.,  8., -9.],
                           [ 2.,  5.,  7.]])

        self.Y = np.array([[-3., -8.],
                           [ 0.,  1.],
                           [ 3.,  2.],
                           [-1.,  0.],
                           [-7., -9.],
                           [ 2.,  9.]])

        self.batch_size = 2

    def test_shapes(self):

        generator = batch_generator(self.X, self.Y, self.batch_size)

        for _ in range(3):

            X_batch, Y_batch = next(generator)

            self.assertEqual(X_batch.shape, (self.batch_size, self.in_features))
            self.assertEqual(Y_batch.shape, (self.batch_size, self.out_features))

    def test_values(self):

        generator = batch_generator(self.X, self.Y, self.batch_size)

        for i in range(3):

            X_batch, Y_batch = next(generator)

            self.assertTrue(np.allclose(X_batch, self.X[i*self.batch_size:(i+1)*self.batch_size, :]))
            self.assertTrue(np.allclose(Y_batch, self.Y[i*self.batch_size:(i+1)*self.batch_size, :]))

    def test_shapes_repeat_after_epoch(self):

        generator = batch_generator(self.X, self.Y, self.batch_size)

        for _ in range(3):
            next(generator)

        for _ in range(3):

            X_batch, Y_batch = next(generator)

            self.assertEqual(X_batch.shape, (self.batch_size, self.in_features))
            self.assertEqual(Y_batch.shape, (self.batch_size, self.out_features))

    def test_values_repeat_after_epoch(self):

        generator = batch_generator(self.X, self.Y, self.batch_size)

        for _ in range(3):
            next(generator)

        for i in range(3):

            X_batch, Y_batch = next(generator)

            self.assertTrue(np.allclose(X_batch, self.X[i*self.batch_size:(i+1)*self.batch_size, :]))
            self.assertTrue(np.allclose(Y_batch, self.Y[i*self.batch_size:(i+1)*self.batch_size, :]))


class TestBatchGeneratorNotDivisible(TestCase):

    def setUp(self):

        self.n_samples = 6
        self.in_features = 3
        self.out_features = 2

        self.X = np.array([[-3.,  0., -8.],
                           [ 0., -2.,  3.],
                           [ 3., -3., -1.],
                           [-1.,  6.,  1.],
                           [-7.,  8., -9.],
                           [ 2.,  5.,  7.]])

        self.Y = np.array([[-3., -8.],
                           [ 0.,  1.],
                           [ 3.,  2.],
                           [-1.,  0.],
                           [-7., -9.],
                           [ 2.,  9.]])

        self.batch_size = 4

    def test_shapes(self):

        generator = batch_generator(self.X, self.Y, self.batch_size)

        # Complete batch
        X_batch, Y_batch = next(generator)

        self.assertEqual(X_batch.shape, (self.batch_size, self.in_features))
        self.assertEqual(Y_batch.shape, (self.batch_size, self.out_features))

        # Incomplete batch
        X_batch, Y_batch = next(generator)

        self.assertEqual(X_batch.shape, (2, self.in_features))
        self.assertEqual(Y_batch.shape, (2, self.out_features))

    def test_values(self):

        generator = batch_generator(self.X, self.Y, self.batch_size)

        # Complete batch
        X_batch, Y_batch = next(generator)

        self.assertTrue(np.allclose(X_batch, self.X[:self.batch_size, :]))
        self.assertTrue(np.allclose(Y_batch, self.Y[:self.batch_size, :]))

        # Incomplete batch
        X_batch, Y_batch = next(generator)

        self.assertTrue(np.allclose(X_batch, self.X[self.batch_size:, :]))
        self.assertTrue(np.allclose(Y_batch, self.Y[self.batch_size:, :]))

    def test_shapes_repeat_after_epoch(self):

        generator = batch_generator(self.X, self.Y, self.batch_size)

        # First epoch
        for _ in range(2):
            next(generator)

        # Second epoch
        # Complete batch
        X_batch, Y_batch = next(generator)

        self.assertEqual(X_batch.shape, (self.batch_size, self.in_features))
        self.assertEqual(Y_batch.shape, (self.batch_size, self.out_features))

        # Incomplete batch
        X_batch, Y_batch = next(generator)

        self.assertEqual(X_batch.shape, (2, self.in_features))
        self.assertEqual(Y_batch.shape, (2, self.out_features))

    def test_values_repeat_after_epoch(self):

        generator = batch_generator(self.X, self.Y, self.batch_size)

        # First epoch
        for _ in range(2):
            next(generator)

        # Second epoch
        # Complete batch
        X_batch, Y_batch = next(generator)

        self.assertTrue(np.allclose(X_batch, self.X[:self.batch_size, :]))
        self.assertTrue(np.allclose(Y_batch, self.Y[:self.batch_size, :]))

        # Incomplete batch
        X_batch, Y_batch = next(generator)

        self.assertTrue(np.allclose(X_batch, self.X[self.batch_size:, :]))
        self.assertTrue(np.allclose(Y_batch, self.Y[self.batch_size:, :]))

class TestCreateModel(TestCase):

    def setUp(self):

        self.in_features = 2
        self.hidden_units = [256, 64, 16, 4]
        self.out_features = 1

    def test_no_exception(self):

        try:
            create_model(self.in_features, self.hidden_units, self.out_features)
        except:
            self.fail('Unexpected error: ' + str(sys.exc_info()))


class TestCompileModel(TestCase):

    def setUp(self):

        self.model = Sequential([
            InputLayer(input_shape=(10,)),
            Dense(256),
            Dense(10)
        ])

        self.optimizer = 'Adam'
        self.learning_rate = 0.001
        self.loss = 'mean_squared_error'

    def test_no_exception(self):

        try:
            compile_model(self.model, self.optimizer, self.learning_rate, self.loss)
        except:
            self.fail('Unexpected error: ' + str(sys.exc_info()))
