import numpy as np
import tensorflow as tf
import pandas as pd


LOSS = "mean_squared_error"
DEFAULT_N_HIDDEN = 1
DEFAULT_N_UNITS = 32
DEFAULT_LAMBDA = 1e-2
DEFAULT_LEARNING_RATE = 1e-2
DEFAULT_N_FILTERS = 32
DEFAULT_KERNEL_SIZE = 1


def build_model_fcnn(n_hidden: int = DEFAULT_N_HIDDEN,
                     n_units: int = DEFAULT_N_UNITS,
                     lambda_: float = DEFAULT_LAMBDA,
                     learning_rate: float = DEFAULT_LEARNING_RATE) -> tf.keras.models.Sequential:
    '''
    Function to build FCNN model according to input parameters.

    :param n_hidden: number of hidden layers
    :param n_units: number of units in each hidden layer
    :param lambda_: L2 regularization parameter
    :param learning_rate: learning rate parameter
    :return: FCNN model
    '''
    model = tf.keras.models.Sequential()
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Dense(n_units,
                                        activation="relu",
                                        kernel_regularizer=tf.keras.regularizers.l2(lambda_)))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss=LOSS, optimizer=optimizer)
    return model


def build_model_cnn(n_hidden: int = DEFAULT_N_HIDDEN,
                    n_units: int = DEFAULT_N_UNITS,
                    lambda_: float = DEFAULT_LAMBDA,
                    learning_rate: float = DEFAULT_LEARNING_RATE,
                    kernel_size: int = DEFAULT_KERNEL_SIZE,
                    ) -> tf.keras.models.Sequential:
    '''
    Function to build CNN model according to input parameters.

    :param n_hidden: number of hidden layers
    :param n_units: number of units in each dense layer & number of filters for conv1D layer
    :param lambda_: L2 regularization parameter
    :param learning_rate: learning rate parameter
    :param kernel_size: size of convolutional kernel
    :return: CNN model
    '''
    model = tf.keras.models.Sequential()
    for layer in range(n_hidden):
        model.add(tf.keras.layers.Conv1D(filters=n_units,
                               kernel_size=(kernel_size,),
                               padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(lambda_)))
    model.add(tf.keras.layers.Dense(units=n_units, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss=LOSS, optimizer=optimizer)
    return model


def build_model_rnn(n_hidden: int = DEFAULT_N_HIDDEN,
                    n_units: int = DEFAULT_N_UNITS,
                    lambda_: float = DEFAULT_LAMBDA,
                    learning_rate: float = DEFAULT_LEARNING_RATE
                    ) -> tf.keras.models.Sequential:
    '''
    Function to build RNN model according to input parameters.

    :param n_hidden: number of hidden layers
    :param n_units: number of units in each hidden layer
    :param lambda_: L2 regularization parameter
    :param learning_rate: learning rate parameter
    :return: RNN model
    '''
    model = tf.keras.models.Sequential()
    for layer in range(n_hidden):
        model.add(tf.keras.layers.LSTM(n_units,
                             return_sequences=True,
                             kernel_regularizer=tf.keras.regularizers.l2(lambda_)))
    model.add(tf.keras.layers.Dense(1))
    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
    model.compile(loss=LOSS, optimizer=optimizer)
    return model

