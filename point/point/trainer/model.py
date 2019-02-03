from keras.models import Sequential
from keras.layers import InputLayer, Dropout, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import numpy as np
from math import ceil


def create_model(in_features, hidden_units, out_features,
                 hidden_activation='relu', output_activation='linear',
                 input_dropout=0.0, hidden_dropout=0.0):
    """Creates a model architecture (uncompiled).

    Args:
        in_features (int): Number of input features.
        hidden_units (list of int): Number of hidden units per layer.
        out_features (int): Number of output features.
        hidden_activation (str): Name of activation for hidden layers.
        output_activation (str): Name of activation for output layer.
        input_dropout (float): Percent of dropout for inputs.
        hidden_dropout (float): Percent of dropout after hidden layers.
    
    Returns:
        Sequential: Keras model.

    """
    model = Sequential()

    model.add(InputLayer(input_shape=(in_features,)))
    model.add(Dropout(rate=input_dropout))

    for hidden_unit in hidden_units:

        model.add(Dense(units=hidden_unit, activation=hidden_activation))
        model.add(Dropout(rate=hidden_dropout))

    model.add(Dense(units=out_features, activation=output_activation))

    return model


def compile_model(model, optimizer, learning_rate, loss, momentum=0.0,
                  decay=0.0, schedule_decay=0.004, nesterov=False,
                  amsgrad=False):
    """Compiles a model for training/evaluation.

    Args:
        model (Sequential): Keras model to compile.
        optimizer (str): Optimizer for learning. Can be `SGD`, `RMSprop`,
            `Adagrad`, `Adadelta`, `Adam`, `Adamax`, or `Nadam`.
        learning_rate (float): Optimizer learning rate.
        loss (str): Loss function metric.
        momentum (float): Momentum for optimizer `SGD`.
        decay (float): Decay for optimizers `SGD`, `RMSprop`, `Adagrad`,
            `Adadelta`, `Adam`, `Adamax`.
        schedule_decay (float): Schedule decay for optimizer `Nadam`.
        nesterov (bool): Whether to use Nesterov momentum for `SGD`.
        amsgrad (bool): Whether to apply AMSGrad variant for `Adam`.

    """
    optimizers = {
        'SGD': SGD(lr=learning_rate, momentum=momentum, decay=decay,
                   nesterov=nesterov),
        'RMSprop': RMSprop(lr=learning_rate, decay=decay),
        'Adagrad': Adagrad(lr=learning_rate, decay=decay),
        'Adadelta': Adadelta(lr=learning_rate, decay=decay),
        'Adam': Adam(lr=learning_rate, decay=decay, amsgrad=amsgrad),
        'Adamax': Adamax(lr=learning_rate, decay=decay),
        'Nadam': Nadam(lr=learning_rate, schedule_decay=schedule_decay)
    }

    model.compile(optimizer=optimizers[optimizer], loss=loss)


def batch_generator(X, Y, batch_size):
    """Generates batches for training and evaluation. If batch_size does not
    evenly divide the data, the last batch will be smaller than batch_size.

    Args:
        X (ndarray): Inputs to break into batches (n_samples, in_features)
        Y (ndarray): Targets to break into batches (n_samples, out_features)
    
    Returns:
        ndarray: Input data in a single batch (batch_size, in_features)
        ndarray: Target data in a single batch (batch_size, out_features)

    """
    n_samples, _ = X.shape

    num_batches = ceil(n_samples / batch_size)

    while True:

        for i in range(num_batches):

            start_index = i * batch_size
            end_index = min(n_samples, (i + 1) * batch_size)

            yield X[start_index:end_index], Y[start_index:end_index]
