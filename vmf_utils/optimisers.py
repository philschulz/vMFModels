import numpy as np
from typing import List, Optional, Callable
from abc import ABC, abstractmethod

DEFAULT_LEARNING_RATE = 0.01
DEFAULT_DECAY_RATE = 0.95
ETA = np.exp(-6)


def compute_rmse(x: np.array) -> np.array:
    """
    Compute the Root Mean Square Error of the input.
    :param x: The input.
    :return: The RMSE.
    """
    return np.sqrt(x + ETA)


def clip_gradient(grad: np.array, threshold: float) -> np.array:
    """
    Normalise and scale gradient if its norm exceeds the threshold.

    :param grad: The gradient matrix.
    :param threshold: The treshold.
    :return: The clipped gradients.
    """
    norms = np.linalg.norm(grad, axis=grad.ndim - 1)
    idx = norms > threshold
    grad[idx] = threshold * grad[idx] / norms[idx].reshape(norms[idx].size, 1)

    return grad


def get_optimiser(optimiser: str,
                  learning_rate: Optional[float] =  DEFAULT_LEARNING_RATE,
                  decay_rate: Optional[float] = DEFAULT_DECAY_RATE,
                  gradient_clipping: Optional[float] = 0) -> Callable:

    if optimiser == "sgd":
        return lambda x: SGD(learning_rate, decay_rate, gradient_clipping)
    elif optimiser == "adagrad":
        return lambda x: AdaGrad(x, learning_rate, gradient_clipping)
    elif optimiser == "adadelta":
        return lambda x: AdaDelta(x, learning_rate, decay_rate, gradient_clipping)


class Optimiser(ABC):
    """
    Abstract class for optimisers.

    :param learning_rate: The (initial) learning rate.
    :param gradient_clipping: Threshold to which larger gradients are scaled. (Not used by default).
    """

    def __init__(self, learning_rate: float =  DEFAULT_LEARNING_RATE, gradient_clipping: Optional[float] = 0):
        self.learning_rate = learning_rate
        self.gradient_clipping = gradient_clipping

    @abstractmethod
    def __call__(self, *gradients: np.array) -> List[np.array]:
        """
        Computes the update given the current gradients.

        :param gradients: The current gradients.
        :return: Updates in the direction of the gradients.
        """
        pass


class SGD(Optimiser):
    """
    Implements SGD with a decaying learning rate.

    :param learning_rate: The (initial) learning rate.
    :param decay_rate: Rate at which the learning rate decays after each update.
    :param gradient_clipping: Threshold to which larger gradients are scaled. (Not used by default).
    """

    def __init__(self, learning_rate: float = DEFAULT_LEARNING_RATE, decay_rate: float = 0.98,
                 gradient_clipping: Optional[float] = 0):
        super().__init__(learning_rate, gradient_clipping)
        self.decay_rate = decay_rate

    def __call__(self, *gradients: np.array) -> List[np.array]:
        """
        Computes the update given the current gradients.

        :param gradients: The current gradients.
        :return: Updates in the direction of the gradients.
        """
        updates = list()

        for idx, grad in enumerate(gradients):
            if self.gradient_clipping > 0:
                grad = clip_gradient(grad, self.gradient_clipping)
            update = self.learning_rate * grad
            updates.append(update)

        self.learning_rate *= self.decay_rate

        return updates


class AdaGrad(Optimiser):
    """
    Implements the AdaGrad optimiser that guarantees convergence by dividing the gradient by its running sum
    of squares.

    :param params: A list of parameter matrices.
    :param learning_rate: The (initial) learning rate.
    :param gradient_clipping: Threshold to which larger gradients are scaled. (Not used by default).
    """

    def __init__(self, params: List[np.array], learning_rate: float = DEFAULT_LEARNING_RATE,
                 gradient_clipping: Optional[float] = 0):
        super().__init__(learning_rate, gradient_clipping)
        self.gradient_history = [np.zeros(param.shape) for param in params]

    def __call__(self, *gradients: np.array) -> List[np.array]:
        """
        Computes the update given the current gradients.

        :param gradients: The current gradients.
        :return: Update in the direction of the gradients.
        """
        updates = list()

        for idx, grad in enumerate(gradients):
            if self.gradient_clipping > 0:
                grad = clip_gradient(grad, self.gradient_clipping)
            self.gradient_history[idx] += np.square(grad)
            update = self.learning_rate * grad / compute_rmse(self.gradient_history[idx])
            updates.append(update)

        return updates


class AdaDelta(AdaGrad):
    """
    Implements the AdaDelta optimiser.

    :param params: A list of parameter matrices.
    :param learning_rate: The (initial) learning rate.
    :param decay_rate: Rate at which the learning rate decays after each update.
    :param gradient_clipping: Threshold to which larger gradients are scaled. (Not used by default).
    """

    def __init__(self, params: List[np.array], learning_rate: float = DEFAULT_LEARNING_RATE, decay_rate: float = 0.95,
                 gradient_clipping: Optional[float] = 0):
        super().__init__(params, learning_rate=learning_rate, gradient_clipping=gradient_clipping)
        self.update_history = [np.zeros(param.shape) for param in params]
        self.decay_rate = decay_rate
        self.init = False

    def __call__(self, *gradients: np.array) -> List[np.array]:
        """
        Computes the update given the current gradients.

        :param gradients: The current gradients.
        :return: Updates in the direction of the gradients.
        """
        updates = list()

        for idx, grad in enumerate(gradients):
            if self.gradient_clipping > 0:
                grad = clip_gradient(grad, self.gradient_clipping)
            learning_rate = compute_rmse(self.update_history[idx]) if self.init else self.learning_rate
            self.gradient_history[idx] = self.decay_rate * self.gradient_history[idx] + (1 - self.decay_rate) * np.square(
                grad)
            update = learning_rate * grad / compute_rmse(self.gradient_history[idx])
            self.update_history[idx] = self.decay_rate * self.update_history[idx] + (1 - self.decay_rate) * np.square(update)
            updates.append(update)

        self.init = True
        return updates
