from scipy.special import iv as bessel
import numpy as np
from vmf_utils import functions as f
from typing import Tuple


class VMF(object):
    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.order = dim /2 - 1
        self.bessel = lambda x: bessel(self.order, x)

    def _normaliser(self, kappa: np.array) -> Tuple[np.array, np.array]:
        log_kappa, log_kappa_grad = f.log_op(kappa)
        log_kappa *= self.order
        log_kappa_grad *= self.order
        log_besssel, log_bessel_grad = self.log_bessel(kappa, self.order)
        norm = log_kappa - log_besssel - (self.order + 1) * np.log(2*np.pi)
        grad = log_kappa_grad - log_bessel_grad

        return norm, grad


    def log_density(self, data: np.array, mu: np.array, kappa: np.array) -> Tuple[np.array, np.array]:
        '''
        Compute the vMF log-likelihood.

        :param data: A data batch. Shape: (batch_size, embed_dim).
        :param mu: The vMF mean directions.
        :param kappa: The vMF concentrations.
        :return: Density and kappa gradients. Shape of both: (batch_size, 1)
        '''
        data_points = data.shape[0]
        norm, norm_grad = self._normaliser(kappa)
        energy = np.dot(mu, data.T)
        density = norm + energy * kappa
        kappa_grad = norm_grad + energy
        mu_grad = kappa * energy

        return density, mu_grad, kappa_grad


    def log_bessel(self, x: np.array, order: int) -> Tuple[np.array, np.array]:
        bessel, bessel_grad = f.bessel_op(x, order)
        log_bessel, log_grad = f.log_op(bessel)

        return log_bessel, bessel_grad * log_grad
