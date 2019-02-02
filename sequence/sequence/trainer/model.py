from keras.models import Sequential, clone_model
from keras.layers import InputLayer, SimpleRNN, GRU, LSTM, TimeDistributed, Dense
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from sequence.trainer.utils import pad


def create_model(batch_input_shape, hidden_units, target_dim,
                 hidden_layer='LSTM', hidden_activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 output_activation='linear', hidden_dropout=0.0,
                 recurrent_dropout=0.0):
    """Creates a model architecture.

    Args:
        batch_input_shape (tuple of int): Input shape for architecture (n_samples,
            batch_length, in_features).
        hidden_units (list of int): Number of units in hidden layers.
        target_dim (tuple of int): Target shape for architecture (n_samples,
            batch_length, out_features)
        hidden_layer (str): Which RNN layers to use. Can be `SimpleRNN`,
            `GRU`, and `LSTM`.
        hidden_activation (str): Activation for hidden layers.
        recurrent_activation (str): Activation for recurrent layers `GRU` and `LSTM`.
        output_activation (str): Activation for outputs.
        hidden_dropout (float): Dropout in hidden layers.
        recurrent_dropout (float): Dropout in recurrent layers.

    Returns:
        Sequential: Keras model ready to be compiled.

    """
    model = Sequential()

    model.add(InputLayer(batch_input_shape=batch_input_shape))

    hidden_layers = {
        'SimpleRNN': lambda units: SimpleRNN(units=units,
                                             activation=hidden_activation,
                                             dropout=hidden_dropout,
                                             recurrent_dropout=recurrent_dropout,
                                             return_sequences=True,
                                             stateful=True),
        'GRU': lambda units: GRU(units=units,
                                 activation=hidden_activation,
                                 recurrent_activation=recurrent_activation,
                                 dropout=hidden_dropout,
                                 recurrent_dropout=recurrent_dropout,
                                 return_sequences=True,
                                 stateful=True),
        'LSTM': lambda units: LSTM(units=units,
                                   activation=hidden_activation,
                                   recurrent_activation=recurrent_activation,
                                   dropout=hidden_dropout,
                                   recurrent_dropout=recurrent_dropout,
                                   return_sequences=True,
                                   stateful=True)
    }

    for hidden_unit in hidden_units:
        model.add(hidden_layers[hidden_layer](hidden_unit))

    model.add(TimeDistributed(
        Dense(units=target_dim[2], activation=output_activation)))

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


def copy_compile_model(model, batch_input_shape):
    """Copies and compiles an existing model with a new input_shape.

    Args:
        model (Sequential): Keras model to copy.
        batch_input_shape (tuple of int): New input shape for copied model.
    
    Returns:
        Sequential: Copied Keras model with new input shape.

    """
    # Copy the structure of the existing model
    copied_model = clone_model(model)

    # Build the model explicitly with new inputs
    copied_model.build(batch_input_shape)

    # Copy the existing model's weights
    copied_model.set_weights(model.get_weights())

    # Compile the model with the existing model's optimizer and loss
    copied_model.compile(model.optimizer, model.loss)

    return copied_model


def batch_generator(X, Y, batch_size=100, pad_val=-1.):
    """Generates batches of sequential data for training and evaluation.
    Splits along axis=1. If batch_size does not evenly divide max_timesteps,
    the data is padded.

    Args:
        X (ndarray): Inputs to break into batches (n_samples, max_timesteps, features).
        Y (ndarray): Targets to break into batches (n_samples, max_timesteps, features).
        batch_size (int): Number of timesteps to include in batches.
        pad_val (float): Pad value to ensure batch_size evenly divides.

    Yields:
        ndarray: Input data in a single batch (n_samples, batch_size, features).
        ndarray: Target data in a single batch (n_samples, batch_size, features).

    """
    _, max_timesteps, _ = X.shape

    if max_timesteps % batch_size != 0:
        raise ValueError(str(max_timesteps) + ' timesteps are not evenly divisible into ' + str(batch_size) + ' batch size.')

    while True:

        for i in range(0, max_timesteps, batch_size):
            yield X[:, i:i+batch_size, :], Y[:, i:i+batch_size, :]
