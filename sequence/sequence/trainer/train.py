import argparse
from sequence.trainer.model import create_model, compile_model, batch_generator, copy_compile_model
from sequence.trainer.utils import load_data, scale, pad, deduce_look_back, replace_expected_look_back, update_prediction_history
from keras.callbacks import Callback, TensorBoard
import keras.backend as backend
import keras.losses as losses
from math import ceil
from os import path, makedirs, remove
from shutil import rmtree
from tempfile import mkdtemp
from sequence.utils import copy_to_gcs
import numpy


class Evaluate(Callback):

    def __init__(self, filepath, X, Y, n_look_back_features=0, look_back_length=None, 
                 batch_size=100, epochs=1, steps=None, frequency=5):
        """Initialize callback for continuous evaluation.
        
        Args:
            filepath (str): Local or remote file name for evaluation summary.
            X (ndarray): Test inputs (n_samples, max_timesteps, features).
            Y (ndarray): Test targets (n_samples, max_timesteps, features).
            n_look_back_features (int): Number of output features used as
                input features. By default, no look back is assumed and
                batch evaluation is used.
            look_back_length (int): Number of historical prediction outputs
                to be used as inputs. By default, no look back is assumed
                and batch evaluation is used.
            batch_size (int): Timestep batch size for test steps. If look back
                is provided, this is ignored.
            epochs (int): Number of epochs to evaluate.
            steps (int): Number of evaluation steps per epoch. If `None`,
                ceil(max_timesteps / batch_size). If look back is provided,
                this is ignored.
            frequency (int): Perform one evaluation every N epochs.

        """
        self.filepath = filepath
        self.filename = path.basename(self.filepath)
        if self.filepath.startswith('gs://'):
            self.dirname = mkdtemp()
        else:
            self.dirname = path.dirname(self.filepath)

        self.X = X
        self.Y = Y
        self.n_look_back_features = n_look_back_features
        self.look_back_length = look_back_length
        self.batch_size = batch_size
        self.epochs = epochs
        if steps is None:
            _, max_timesteps, _ = self.X.shape
            self.steps = ceil(max_timesteps / batch_size)
        else:
            self.steps = steps
        self.frequency = frequency

        # Initialize the summary header
        with open(path.join(self.dirname, self.filename), 'w') as eval_summary:
            eval_summary.write('epoch,loss\n')
        
        # Copy to GCS if remote
        if self.filepath.startswith('gs://'):

            local_file = path.join(self.dirname, self.filename)
            remote_file = self.filepath

            copy_to_gcs(local_file, remote_file)

    def on_epoch_end(self, epoch, logs=None):
        """Triggers after each epoch. If frequency epochs reached, evaluation
        will occur.

        Args:
            epoch (int): Index of epoch
            logs (dict of str: float): Metric results for this training epoch.

        """
        if (epoch + 1) % self.frequency == 0:

            # Evaluate the model
            print('Evaluating')

            # If there are look back features, we have to wait for the output for each 
            # timestep, hence the outer loop is required.
            if self.n_look_back_features > 0:

                # Create the evaluation model from existing model with new input shape
                n_samples, _, in_features = self.X.shape
                eval_model = copy_compile_model(self.model, (n_samples, 1, in_features))

                Y_pred = numpy.zeros(self.Y.shape)
                Y_pred_hist = numpy.zeros((self.Y.shape[0], self.look_back_length, self.Y.shape[2]))

                for t in range(self.X.shape[1]):

                    X_t = replace_expected_look_back(self.X[:, t:t + 1, :], Y_pred_hist, self.n_look_back_features)

                    Y_pred[:, t:t + 1, :] = eval_model.predict(X_t)

                    Y_pred_hist = update_prediction_history(Y_pred_hist, Y_pred[:, t:t + 1, :])
                
                # Compute the loss manually.
                loss = numpy.mean(backend.eval(losses.get(eval_model.loss)(self.Y, Y_pred)))
            
            # If there are no look back features, batch estimation is possible and
            # no outer loop is necessary.
            else:

                # Create the evaluation model from existing model with new input shape
                n_samples, _, in_features = self.X.shape
                eval_model = copy_compile_model(self.model, (n_samples, self.batch_size, in_features))

                loss = eval_model.evaluate_generator(batch_generator(self.X, self.Y, self.batch_size),
                                                     steps=self.epochs * self.steps, verbose=1)
            
            # Record metric
            with open(path.join(self.dirname, self.filename), 'a') as eval_summary:
                eval_summary.write('{},{}\n'.format(epoch + 1, loss))
            
            # Copy to GCS if remote
            if self.filepath.startswith('gs://'):
                
                local_file = path.join(self.dirname, self.filename)
                remote_file = self.filepath

                copy_to_gcs(local_file, remote_file)

    def on_train_end(self, logs=None):
        """Triggered after all training completes. Cleans up temporary
        directory (if remote logging).

        Args:
            logs (dict of str: float): Currently nothing is passed.

        """
        if self.filepath.startswith('gs://'):
            rmtree(self.dirname)


class ModelCheckpoint(Callback):
    """More limited model checkpoint, but supports GCS saving."""

    def __init__(self, filepath, save_weights_only=False, period=1):
        """Initialize model checkpoint.

        Args:
            filepath (str): Local or remote file name for checkpoints.
            save_weights_only (bool): Whether to just save weights.
            period (int): Save checkpoint every N epochs.

        """
        self.filepath = filepath
        self.filename = path.basename(filepath)
        if self.filepath.startswith('gs://'):
            self.dirname = mkdtemp()
        else:
            self.dirname = path.dirname(filepath)

        self.save_weights_only = save_weights_only
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        """Triggers after every epoch. If period is reached, model will be
        saved.

        Args:
            epoch (int): Current epoch (0-index).
            logs (dict of str: float): Metrics for current epoch.

        """
        if (epoch + 1) % self.period == 0:

            logs['epoch'] = epoch + 1
            local_file = path.join(self.dirname, self.filename.format(**logs))

            if self.save_weights_only:
                self.model.save_weights(local_file)
            else:
                self.model.save(local_file)
            
            if self.filepath.startswith('gs://'):

                remote_file = self.filepath.format(**logs)

                copy_to_gcs(local_file, remote_file)
                remove(local_file)


class CSVLogger(Callback):
    """CSVLogger supporting GCS storage."""

    def __init__(self, filepath):
        """Initialize CSV training log callback.

        Args:
            filepath (str): Filename of log file.

        """
        self.filepath = filepath
        self.filename = path.basename(self.filepath)
        if self.filepath.startswith('gs://'):
            self.dirname = mkdtemp()
        else:
            self.dirname = path.dirname(self.filepath)

        # Intialize summary header
        with open(path.join(self.dirname, self.filename), 'w') as train_summary:
            train_summary.write('epoch,loss\n')
        
        # Copy to GCS is remote
        if self.filepath.startswith('gs://'):

            local_file = path.join(self.dirname, self.filename)
            remote_file = self.filepath

            copy_to_gcs(local_file, remote_file)
    
    def on_epoch_end(self, epoch, logs=None):
        """Triggers after each epoch. Saves epoch metrics.

        Args:
            epoch (int): Index of epoch
            logs (dict of str: float): Metric results for this training epoch.

        """
        # Log current metrics
        with open(path.join(self.dirname, self.filename), 'a') as train_summary:
            train_summary.write('{},{}\n'.format(epoch + 1, logs['loss']))
        
        # Copy to GCS if remote
        if self.filepath.startswith('gs://'):

            local_file = path.join(self.dirname, self.filename)
            remote_file = self.filepath

            copy_to_gcs(local_file, remote_file) 

    def on_train_end(self, logs=None):
        """Triggered after all training completes. Cleans up temporary
        directory (if remote logging).

        Args:
            logs (dict of str: float): Currently nothing is passed.

        """
        if self.filepath.startswith('gs://'):
            rmtree(self.dirname)


def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, n_look_back_features,
                       look_back_length, job_dir, train_batch_size=100,
                       eval_batch_size=100, train_epochs=200, eval_epochs=1,
                       train_steps=None, eval_steps=None, eval_frequency=5,
                       checkpoint_frequency=5):
    """Train an existing model with specified training data. Continuously
    evaluate model, save model, and log epoch results.

    Args:
        model (Sequential): Keras model to train and evaluate.
        X_train (ndarray): Training inputs (n_samples, max_timesteps, features).
        Y_train (ndarray): Training targets (n_samples, max_timesteps, features).
        X_test (ndarray): Test inputs (n_samples, max_timesteps, features).
        Y_test (ndarray): Test targets (n_samples, max_timesteps, features).
        n_look_back_features (int):
        look_back_length (int):
        job_dir (str): Directory to save model checkpoints and scores.
        train_batch_size (int): Timestep batch size for training steps.
        eval_batch_size (int): Timestep batch size for test steps.
        train_epochs (int): Number of epochs to train.
        eval_epochs (int): Number of epochs to evaluate.
        train_steps (int): Number of training steps per epoch. If `None`,
            ceil(train_max_timesteps / train_batch_size).
        eval_steps (int): Number of evaluation steps per epoch. If `None`,
            ceil(eval_max_timesteps / eval_batch_size).
        eval_frequency (int): Perform one evaluation every N epochs.
        checkpoint_frequency (int): Perform one checkpoint every N epochs.
    
    """
    # Initialize callbacks
    if not job_dir.startswith('gs://'):
        makedirs(path.join(job_dir, 'weights'), exist_ok=True)
    checkpoint_filename = path.join(job_dir, 'weights', 'E{epoch:03d}L{loss:.6E}.hdf5')
    checkpoint_callback = ModelCheckpoint(checkpoint_filename,
                                          period=checkpoint_frequency,
                                          save_weights_only=True)
    
    if not job_dir.startswith('gs://'):
        makedirs(path.join(job_dir, 'logs'), exist_ok=True)
    eval_summary_filename = path.join(job_dir, 'logs', 'evaluation_results.csv')
    evaluate_callback = Evaluate(eval_summary_filename, X_test, Y_test,
                                 n_look_back_features, look_back_length,
                                 eval_batch_size, eval_epochs, eval_steps,
                                 eval_frequency)

    train_summary_filename = path.join(job_dir, 'logs', 'training_results.csv')
    logging_callback = CSVLogger(train_summary_filename)
    
    callbacks = [logging_callback, checkpoint_callback, evaluate_callback]

    # Start the training loop
    if not train_steps:

        _, max_timesteps, _ = X_train.shape
        train_steps = ceil(max_timesteps / train_batch_size)

    model.fit_generator(batch_generator(X_train, Y_train, train_batch_size),
                        steps_per_epoch=train_steps, epochs=train_epochs,
                        callbacks=callbacks, shuffle=False)


def main(job_dir, train_file, eval_file, first_layer_size=256, num_layers=1,
         scale_factor=0.25, optimizer='RMSprop', learning_rate=0.001,
         loss='mean_squared_error', train_batch_size=100,
         eval_batch_size=100, train_epochs=200, eval_epochs=1,
         train_steps=None, eval_steps=None, eval_frequency=5,
         checkpoint_frequency=5):
    """Create a model. Train and evaluate.

    Args:
        job_dir (str): Directory to save model checkpoints and scores.
        train_file (str): Training data.
        eval_file (str): Evaluation data.
        first_layer_size (int): Number of units in first layer.
        num_layers (int): Number of hidden layers.
        scale_factor (float): Rate of decay size of units in hidden layers.
            max(1, int(first_layer_size * scale_factor ** i))
        optimizer (str): Optimizer for training.
        learning_rate (float): Learning rate for optimizer.
        loss (str): Loss for training and evaluation.
        train_batch_size (int): Timestep batch size for training steps.
        eval_batch_size (int): Timestep batch size for test steps.
        train_epochs (int): Number of epochs to train.
        eval_epochs (int): Number of epochs to evaluate.
        train_steps (int): Number of training steps per epoch. If `None`,
            ceil(train_max_timesteps / train_batch_size).
        eval_steps (int): Number of evaluation steps per epoch. If `None`,
            ceil(eval_max_timesteps / eval_batch_size).
        eval_frequency (int): Perform one evaluation every N epochs.
        checkpoint_frequency (int): Perform one checkpoint every N epochs.

    """
    # Load training data
    X_train, in_features, Y_train, out_features = load_data(train_file, pad_val=-1.)

    X_train_padded = pad(X_train, train_batch_size, pad_val=-1.)
    Y_train_padded = pad(Y_train, train_batch_size, pad_val=-1.)

    X_train_scaled, _, _ = scale(X_train_padded, min=0., max=1., pad_old=-1.)
    Y_train_scaled, _, _ = scale(Y_train_padded, min=0., max=1., pad_old=-1.)

    # Load evaluation data
    X_test, _, Y_test, _ = load_data(eval_file, pad_val=-1.)

    X_test_padded = pad(X_test, eval_batch_size, pad_val=-1.)
    Y_test_padded = pad(Y_test, eval_batch_size, pad_val=-1.)

    X_test_scaled, _, _ = scale(X_test_padded, min=0., max=1., pad_old=-1.)
    Y_test_scaled, _, _ = scale(Y_test_padded, min=0., max=1., pad_old=-1.)

    # Configure model
    n_train_samples, _, n_in_features = X_train_scaled.shape
    n_train_samples, _, n_out_features = Y_train_scaled.shape

    hidden_units = [max(1, int(first_layer_size * scale_factor ** i)) for i in range(num_layers)]

    model = create_model(batch_input_shape=(n_train_samples, train_batch_size, n_in_features), 
                         hidden_units=hidden_units,
                         target_dim=(n_train_samples, train_batch_size, n_out_features))
    
    compile_model(model, optimizer, learning_rate, loss)

    # Determine look back properties
    n_look_back_features, look_back_length = deduce_look_back(in_features, out_features)

    # Start train/evaluate loop
    train_and_evaluate(model, X_train_scaled, Y_train_scaled, X_test_scaled,
                       Y_test_scaled, n_look_back_features, look_back_length,
                       job_dir, train_batch_size, eval_batch_size,
                       train_epochs, eval_epochs, train_steps, eval_steps,
                       eval_frequency, checkpoint_frequency)


if __name__ == '__main__':
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--job-dir',
        type=str,
        default='./Jobs/Sequence/001',
        help='Directory to save model checkpoints and scores.')
    parser.add_argument(
        '--train-file',
        type=str,
        default='./Jobs/Sequence/001/data/train.csv',
        help='Training data.')
    parser.add_argument(
        '--eval-file',
        type=str,
        default='./Jobs/Sequence/001/data/test.csv',
        help='Evaluation data.')
    parser.add_argument(
        '--first-layer-size',
        type=int,
        default=256,
        help='Number of units in first layer.')
    parser.add_argument(
        '--num-layers',
        type=int,
        default=1,
        help='Number of hidden layers.')
    parser.add_argument(
        '--scale-factor',
        type=float,
        default=0.25,
        help='Rate of decay size of units in hidden layers. max(1, int(first_layer_size * scale_factor ** i))')
    parser.add_argument(
        '--optimizer',
        type=str,
        default='RMSprop',
        help='Optimizer for training.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='Learning rate for optimizer.')
    parser.add_argument(
        '--loss',
        type=str,
        default='mean_squared_error',
        help='Loss for training and evaluation.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=100,
        help='Timestep batch size for training steps.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=100,
        help='Timestep batch size for test steps. If look back inputs are used, this is ignored.')
    parser.add_argument(
        '--train-epochs',
        type=int,
        default=200,
        help='Number of epochs to train.')
    parser.add_argument(
        '--eval-epochs',
        type=int,
        default=1,
        help='Number of epochs to evaluate.')
    parser.add_argument(
        '--train-steps',
        type=int,
        default=None,
        help='Number of training steps per epoch. If `None`, ceil(train_max_timesteps / train_batch_size).')
    parser.add_argument(
        '--eval-steps',
        type=int,
        default=None,
        help='Number of evaluation steps per epoch. If `None`, ceil(eval_max_timesteps / eval_batch_size). If look back inputs are used, this is ignored.')
    parser.add_argument(
        '--eval-frequency',
        type=int,
        default=5,
        help='Perform one evaluation every N epochs.')
    parser.add_argument(
        '--checkpoint-frequency',
        type=int,
        default=5,
        help='Perform one checkpoint every N epochs.')

    args, _ = parser.parse_known_args()
    main(**vars(args))
