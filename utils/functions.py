import numpy as np
from scipy.special import gammaln, expit, digamma, iv as bessel
from typing import Tuple


sigmoid = expit
soft_plus_derivative = expit


def soft_plus(x: np.array) -> Tuple[np.array, np.array]:
    result = np.log(np.exp(x) + 1)
    gradient = soft_plus_derivative(x)

    return result, gradient


def log_bessel_derivative(order: int, kappa: np.array) -> np.array:
    return 0.5 * (bessel(order - 1, kappa) + bessel(order + 1, kappa)) / bessel(order, kappa)


def gamma_kl(shape_q: np.array, rate_q: np.array, shape_p: np.array, rate_p: np.array) -> np.array:
    expected_value_q = shape_q / rate_q
    expected_log_value_q = digamma(shape_q) - np.log(rate_q)

    log_normalizer_q = np.log(rate_q) * shape_q - gammaln(shape_q)
    log_normalizer_p = np.log(rate_p) * shape_p - gammaln(shape_p)

    return expected_log_value_q * (shape_q - shape_p) + expected_value_q * (
    rate_q - rate_p) + log_normalizer_q - log_normalizer_p


def diagonal_gaussian_kl(mean: np.array, std: np.array) -> float:
    """
    Computes the KL divergence between a diagonal and a standard Gaussian.

    :param mean: The mean of the diagonal Gaussian.
    :param std: Standard deviations of the diagonal Gaussian.
    :return: The KL divergence.
    """
    var=std**2
    return 0.5 * np.sum(1 + np.log(var) - mean ** 2 - var, axis=1)


def diagonal_gaussian_kl_grad(mean: np.array, std: np.array) -> Tuple[float, float]:
    mean_grad = -np.sum(mean, axis=1)
    std_grad = np.sum(std/(std**2) - std)
    return mean_grad, std_grad


def std_gaussian_sample_transform(self, mu: np.array, sigma: np.array):
    epsilon = np.random.norm(sigma.size)
    x = mu + sigma * epsilon
    return x, 1, epsilon