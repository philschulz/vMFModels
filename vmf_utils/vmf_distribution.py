from scipy.special import iv as bessel
import numpy as np
from typing import Tuple


class VMF(object):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.bessel = lambda x: bessel(dim / 2 - 1, x)

    def log_density(self, data: np.array, mu: np.array, kappa: np.array) -> Tuple[np.array, np.array]:
        '''
        Compute the vMF log-likelihood.

        :param data: A data batch. Shape: (batch_size, embed_dim).
        :param mu: The vMF mean directions.
        :param kappa: The vMF concentrations.
        :return: Density and kappa gradients. Shape of both: (batch_size, 1)
        '''
        data_points = data.shape[0]
        normaliser = (np.log(kappa) * (self.dim / 2 - 1) - np.log(self.bessel(kappa))) - (self.dim/2) * np.log(2*np.pi)
        normaliser *= data_points
        energy = np.dot(mu, data.T)
        density = normaliser + energy * kappa
        kappa_grad = (self.dim / 2 - 1) / kappa - self.__log_bessel_gradient(kappa)
        kappa_grad = data_points * kappa_grad + np.sum(energy, axis=energy.ndim - 1, keepdims=True)
        mu_grad = kappa * np.sum(data, axis=data.ndim - 2)

        return density, mu_grad, kappa_grad

    def __log_bessel_gradient(self, kappa: float) -> float:
        return (bessel(self.dim / 2 - 2, kappa) - bessel(self.dim / 2, kappa)) / 2 * self.bessel(kappa)
