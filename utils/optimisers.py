import numpy as np
from typing import List


class AdaGrad(object):
    """
    Implements the AdaGrad optimiser that guarantees convergence by dividing the gradient by its running sum
    of squares.

    :param params: A list of parameters matrices.
    :param learning_rate: An initial learning rate.
    """

    def __init__(self, params: List[np.array], learning_rate: float = 0.001):
        self.gradient_history = [np.zeros(param.shape) for param in params]
        self.learning_rate = learning_rate
        self.eta = np.exp(-8)

    def compute_updates(self, gradients: List[np.array]) -> List[np.array]:
        """
        Computes the udpate given the current gradients.

        :param gradients: The current gradients.
        :return: Update in the direction of the gradients.
        """
        updates = list()

        for idx, grad in enumerate(gradients):
            self.gradient_history[idx] += np.square(grad)
            update = self.learning_rate * grad / (np.sqrt(self.gradient_history[idx]) + self.eta)
            updates.append(update)

        return updates

