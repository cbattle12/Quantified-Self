import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from typing import Callable

WEIGHT_COL_INDEX = -3  # column index of weight
END_FRACTION_TRAINING_DATA = 0.8
END_FRACTION_VALIDATION_DATA = 0.9
EPOCHS = 100
PATIENCE = 10
N_ITER = 50 # number of search iterations
CV = 3 # k-fold cv value
RANDOM_STATE = 42


def rescale_window_split_dataframe(df: pd.DataFrame,
                                   end_fraction_training: float = END_FRACTION_TRAINING_DATA,
                                   end_fraction_val: float = END_FRACTION_VALIDATION_DATA,
                                   n_timesteps: int = 1) -> tuple:
    '''
    Split dataframe into train/validation/test data sets and labels, rescale the data, and window it according to the
    number of timesteps. Data sets and labels are numpy arrays, with datasets of shape (num. rows, num. timesteps,
    num. features). NaN values are ignored and samples containing NaNs are dropped so that all outputted samples are
    NaN free. This leads to the possibility that there are gaps between samples, e.g. that consecutive samples don't
    always belong to consecutive days.

    :param df: dataframe to scale, window, and split
    :param end_fraction_training: end fraction of data to use for training set
    :param end_fraction_val: end fraction of data to use for validation set
    :param n_timesteps: number of time steps to include in each sample
    :return: tuple of training/vlidation/test datasets and labels, as well as the idxs of NaN values in the test set
    '''
    if "Label" not in df.columns:
        raise ValueError("Label column doesn't exist, please check the input dataframe.")

    # Split dataframe into train/val/test sets
    n_rows = len(df)
    train_end_idx = int(end_fraction_training * n_rows)
    val_end_idx = int(end_fraction_val * n_rows)

    df_train = df[0:train_end_idx]
    df_val = df[train_end_idx:val_end_idx]
    df_test = df[val_end_idx:]

    if len(df_train) == 0 or len(df_val) == 0 or len(df_test) == 0:
        raise ValueError("Train, validation, or test set has length of 0. Please check end fraction values.")

    # Rescale by training set mean and std (only train to avoid influence from val/test sets).
    train_mean = df_train.mean()
    train_std = df_train.std()
    df_train = (df_train - train_mean) / train_std
    df_val = (df_val - train_mean) / train_std
    df_test = (df_test - train_mean) / train_std

    # Get rid of NaNs, create label arrays, & convert to numpy arrays
    label_idx = df.columns.get_loc("Label")

    X_train, y_train, _ = _convert_df_to_numpy_no_nans(df_train, n_timesteps, label_idx)
    X_val, y_val, _ = _convert_df_to_numpy_no_nans(df_val, n_timesteps, label_idx)
    X_test, y_test, nan_idxs_test = _convert_df_to_numpy_no_nans(df_test, n_timesteps, label_idx)

    return X_train, y_train, X_val, y_val, X_test, y_test, nan_idxs_test


def run_search(build_model_fnc: Callable,
               params: dict,
               X_train: np.ndarray,
               y_train: np.ndarray,
               X_val: np.ndarray,
               y_val: np.ndarray,
               epochs: int = EPOCHS,
               patience: int = PATIENCE,
               n_iter: int = N_ITER,
               random_state: int = RANDOM_STATE,
               cv: int = CV) -> RandomizedSearchCV:
    '''
    Run randomized search in parameter space given by params for model built by build_model_fnc.

    :param build_model_fnc: function which builds a Keras model
    :param params: dictionary defining parameter space of random search
    :param X_train: training set
    :param y_train: training set labels
    :param X_val: validation set
    :param y_val: validation set labels
    :param epochs: number of epochs run in each search
    :param patience: patience parameter for early stopping
    :param n_iter: number of random search iterations
    :param random_state: random seed for randomized search
    :param cv: number of cross-validation folds performed when assessing model score
    :return: random search object with optimal parameter set
    '''
    keras_reg = tf.keras.wrappers.scikit_learn.KerasRegressor(build_model_fnc)

    search_cv = RandomizedSearchCV(keras_reg, params, n_iter=n_iter, cv=cv, random_state=random_state)
    search_cv.fit(X_train, y_train, epochs=epochs,
                  validation_data=(X_val, y_val),
                  callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience)])
    return search_cv


def fit_and_save_best_model(build_model_fnc: Callable,
                            params: dict,
                            filepath: str,
                            X_train: np.ndarray,
                            y_train: np.ndarray,
                            X_val: np.ndarray,
                            y_val: np.ndarray,
                            epochs: int = EPOCHS,
                            patience: int = PATIENCE) -> None:
    '''
    Fit Keras model given optimal parameter set and save best model (from running early stopping).

    :param build_model_fnc: function which builds a Keras model
    :param params: dictionary of best parameter set
    :param filepath: filepath where model should be saved
    :param X_train: training set
    :param y_train: training set labels
    :param X_val: validation set
    :param y_val: validation set labels
    :param epochs: number of epochs run in each search
    :param patience: patience parameter for early stopping
    :return: None
    '''
    model = build_model_fnc(**params)
    model.fit(X_train, y_train, epochs=epochs,
              validation_data=(X_val, y_val),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience),
                         tf.keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                            monitor="val_loss",
                                                            save_best_only=True)])


def plot_predictions(day_number: np.ndarray,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     y_lin: np.ndarray,
                     y_fcnn: np.ndarray,
                     y_cnn: np.ndarray,
                     y_rnn: np.ndarray,
                     nan_idxs_test: np.ndarray,
                     nan_idxs_cnn: np.ndarray,
                     nan_idxs_rnn: np.ndarray,
                     n_timesteps_cnn: int,
                     n_timesteps_rnn: int,
                     plotting_bounds: tuple) -> None:
    '''
    Helper function to plot various model predictions against each other.

    :param day_number: array of day numbers
    :param X_test: test set
    :param y_test: test set labels
    :param y_lin: linear model predictions
    :param y_fcnn: FCNN model predictions
    :param y_cnn: CNN model predictions
    :param y_rnn: RNN model predictions
    :param nan_idxs_test: indexes of NaN values in test set
    :param nan_idxs_cnn: indexes of NaN values in test set for number of timesteps in CNN
    :param nan_idxs_rnn: indexes of NaN values in test set for number of timesteps in RNN
    :param n_timesteps_cnn: number of timesteps for CNN model
    :param n_timesteps_rnn: number of timesteps for RNN model
    :param plotting_bounds: indexes range to plot the predictions over
    :return: None
    '''
    cnn_offset = n_timesteps_cnn - 1
    rnn_offset = n_timesteps_rnn - 1
    baseline = X_test[:, 0, WEIGHT_COL_INDEX]

    # Expand all arrays with NaN values
    y_test = _add_nans(y_test, nan_idxs_test)
    y_lin = _add_nans(y_lin, nan_idxs_test)
    y_fcnn = _add_nans(y_fcnn, nan_idxs_test)
    y_cnn = _add_nans(y_cnn, nan_idxs_cnn)
    y_rnn = _add_nans(y_rnn, nan_idxs_rnn)
    baseline = _add_nans(baseline, nan_idxs_test)
    day_number = np.arange(day_number[0], day_number[0] + len(y_test))
    s = slice(*plotting_bounds)

    # Offset for multiple time step case
    if cnn_offset:
        y_cnn = np.concatenate((np.array([np.nan] * cnn_offset), y_cnn))
    if rnn_offset:
        y_rnn = np.concatenate((np.array([np.nan] * rnn_offset), y_rnn))

    plt.scatter(day_number[s], y_test[s], c="darkgray", s=70, alpha=0.7, label="Data")
    plt.plot(day_number[s], baseline[s], c="darkgray", linestyle="--", linewidth=2, label="Baseline")
    plt.plot(day_number[s], y_lin[s], linewidth=3, c="dimgray", label="Linear Model")
    plt.plot(day_number[s], y_fcnn[s], linewidth=3, label="FCNN Model")
    plt.plot(day_number[s], y_cnn[s], linewidth=3, label="CNN Model")
    plt.plot(day_number[s], y_rnn[s], linewidth=3, label="RNN Model")
    plt.xlabel("Day Number")
    plt.ylabel("Scaled Weight")
    plt.legend()
    plt.show()


def plot_linear_model_coeffs(model_lin: LinearRegression, X_train: np.ndarray, df: pd.DataFrame) -> None:
    '''
    Helper function to plot linear regression weights.

    :param model_lin: linear regression model
    :param X_train: training set
    :param df: data frame
    :return: None
    '''
    plt.bar(x = range(X_train.shape[-1]), height=model_lin.coef_)
    plt.ylabel("Linear Model Coefficient")
    axis = plt.gca()
    axis.set_xticks(range(X_train.shape[-1]))
    _ = axis.set_xticklabels(df.columns, rotation=45)


def _convert_df_to_numpy_no_nans(df: pd.DataFrame, n_timesteps: int, label_idx: int) -> tuple:
    '''
    Helper function to window data frame, convert it to numpy, extract the label array, and extract its NaN indexes.

    :param df: data frame
    :param n_timesteps: number of timesteps for each sample
    :param label_idx: column index of label column
    :return: tuple of dataset, label, and NaN indexes
    '''
    X = np.asarray([df.to_numpy()[i:i + n_timesteps] for i in range(len(df) - n_timesteps + 1)])
    nan_idxs = [idx for idx, arr in enumerate(X) if np.sum(np.isnan(arr)) > 0]  # used for plotting
    X = np.asarray([arr for arr in X if np.sum(np.isnan(arr)) == 0])  # get rid of NaNs
    y = X[:, :, label_idx]
    X = np.delete(X, label_idx, axis=2)  # drop labels from data array
    return X, y, nan_idxs


def _add_nans(arr: np.ndarray, nan_idxs: np.ndarray) -> np.ndarray:
    '''
    Helper function to add NaN values back into an array in locations specified by nan_idxs. Used for plotting model
    predictions with the original data set spacing.

    :param arr: input array
    :param nan_idxs: indexes where NaNs are to be added
    :return: input array with NaNs added
    '''
    arr = arr.tolist()
    for idx in nan_idxs:
        arr.insert(idx, np.nan)
    return np.array(arr)