import numpy as np
import matplotlib.pyplot as plt
from typing import Union

N_BOOTSTRAP_ITERATIONS = 1000
CI_95_FACTOR = 1.96


def get_probability_1_conditional_1(binary_arr: np.ndarray) -> float:
    '''
    Get empirical probability P(1|1) that the number subsequent to 1 in a binary sequence is 1. Input array must consist
    of only 0s and 1s

    :param binary_arr: array of 0s and 1s
    :return: empirical probability P(1|1)
    '''
    if set(binary_arr) != {0,1}:
        raise ValueError("Input array must consist of only 0 and 1")

    non_zero_idxs = binary_arr.nonzero()[0]
    values_following_ones = [binary_arr[idx+1] for idx in non_zero_idxs if idx < len(binary_arr) - 1]
    return np.sum(values_following_ones) / (len(values_following_ones) - 1)


def get_bootstrap_distribution(x: np.ndarray, n_iter) -> np.ndarray:
    '''
    Create array of n_iter resamples with replacement of x for bootstrapping

    :param x: array to be resampled for bootstrapping
    :param n_iter: number of bootstrap resamples of x
    :return: (n_iter, len(x)) array of resamples of x
    '''
    arr_length = len(x)
    if arr_length > 800:
        # For long arrays this method is faster
        bootstrap_list = []
        for _ in range(n_iter):
            bootstrap_list.append(np.random.choice(x, arr_length))
        bootstrap_x = np.array(bootstrap_list)
    else:
        # This is faster for shorter arrays
        bootstrap_x = np.repeat(x, n_iter)
        bootstrap_x = np.random.choice(bootstrap_x, len(bootstrap_x))
        bootstrap_x = bootstrap_x.reshape(n_iter, arr_length)
    return bootstrap_x


def run_pearson_correlation_bootstrap(x: np.ndarray, y: np.ndarray, n_iter: int=N_BOOTSTRAP_ITERATIONS) -> np.ndarray:
    '''
    Return distribution of Pearson correlation coefficients for n_iter bootstrap resamples of input arrays.

    :param x: 1st input array to correlate
    :param y: 2nd input array to correlate
    :param n_iter: number of bootstrap resamples
    :return: distribution of bootstrapped Pearson correlation coefficient
    '''
    if len(x) != len(y):
        raise ValueError(f"Inputs x and y must have equal lengths. {len(x)} != {len(y)}")

    bootstrap_x = get_bootstrap_distribution(x, n_iter)
    bootstrap_y = get_bootstrap_distribution(y, n_iter)
    return np.array([np.corrcoef(bootstrap_x[i], bootstrap_y[i])[0, 1] for i in range(n_iter)])


def run_probability_1_conditional_1_bootstrap(binary_arr: np.ndarray, n_iter: int=N_BOOTSTRAP_ITERATIONS) -> np.ndarray:
    '''
    Return distribution of empirical probabilities P(1|1) for n_iter bootstrap resamples of input array.

    :param binary_arr: array of 0s and 1s
    :param n_iter: number of bootstrap resamples
    :return: distribution of bootstrapped empirical probabilities P(1|1)
    '''
    bootstrap_arr = get_bootstrap_distribution(binary_arr, n_iter)
    return np.apply_along_axis(get_probability_1_conditional_1, 0, bootstrap_arr)


def find_regions_increasing_slope(smooth_y: np.ndarray) -> list:
    '''
    Return a list of (start_idx, end_idx) bounds between which the slope of y is increasing. IMPORTANT: The input
    variable must be smoothed to get a result which isn't extremely noisy. Cases of increasing slope for only a single
    point are ignored and not included in the output list

    :param smooth_y: smoothed input array
    :return: list of bounds between which the slope of y is increasing
    '''

    dy = np.diff(smooth_y)
    idxs_increasing_slope = []
    idx_holder = []
    for idx, value in enumerate(dy):
        if value > 0 and not idx_holder:
            idx_holder.append(idx)
        elif idx == len(dy) - 1 and idx_holder:
            idxs_increasing_slope.append((idx_holder[0], idx))
        elif value <= 0 and idx_holder:
            if idx_holder[0] + 1 == idx:
                # Ignore regions of length 1 and reset idx holder
                idx_holder = []
            else:
                idxs_increasing_slope.append((idx_holder[0], idx))
                idx_holder = []
    return idxs_increasing_slope


def plot_bootstrap_histogram_with_CI_95(distribution: np.ndarray,
                                        test_statistic_value: Union[float, None] = None,
                                        ci_factor: float = CI_95_FACTOR) -> None:
    '''
    Helper function to plot bootstrapped histogram with confidence intervals and a test statistic value (with the aim to
    see if the test statistic value falls within the confidence interval).

    :param distribution: bootstrap distribtution to plot
    :param test_statistic_value: test statistic value e.g. mean to compare to the bootstrapped distribution
    :param ci_factor: confidence interval multiplier of standard deviation
    :return: None
    '''
    mu = distribution.mean()
    sigma = distribution.std()
    plt.hist(distribution, 30)
    if test_statistic_value is not None:
        plt.axvline(x=test_statistic_value, c="tab:orange", linestyle="--", linewidth=3,
                    label="Test statistic from data")
    plt.gca().axvspan(mu - ci_factor * sigma, mu + ci_factor * sigma, facecolor='gray', alpha=0.3, label="95% CI")
    plt.ylabel("Counts")
    plt.legend()

