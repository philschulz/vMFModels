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
        :return: Density and kappa gradients. Shape of both: (batch_size, 1)
        '''
        normaliser = (np.log(kappa) * (self.dim / 2 - 1) - np.log(self.bessel(kappa)))
        energy = np.dot(mu, data.T)
        density = normaliser + energy * kappa
        kappa_grad = (self.dim / 2 - 1) / kappa - self.__log_bessel_gradient(kappa)
        kappa_grad = data.shape[0] * kappa_grad + np.sum(energy, axis=1)[:,np.newaxis]

        # if np.any(np.isnan(density)):
        #     print(density)
        #     print("kappa = {}".format(kappa))

        return density, kappa_grad

    def __log_bessel_gradient(self, kappa: float) -> float:
        return (bessel(self.dim / 2 - 2, kappa) - bessel(self.dim / 2, kappa)) / self.bessel(kappa)
