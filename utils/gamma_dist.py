from scipy.stats import gamma
from math import log

class GammaDist(object):
    '''Class that implements the gamma distribution with shape-scale parametrisation. The density is
    p(x) = (x**(shape-1) exp(-x/scale))/gamma(shape)*(scale**shape)
    '''

    def __init__(self, shape, scale):
        '''Constructor

        :param shape: The shape parameter of the distribution
        :param scale: The scale parameter of the distribution
        '''

        self.shape = shape
        self.scale = scale

    def density(self, x):
        '''Compute the density a point x (x needs to be positive)

        :param x: A point in (0,infinity)
        :return: The density of this distribution at x
        '''
        return gamma.pdf(x, self.shape, 0, self.scale)

    def log_density(self, x):
        '''Compute the log-density at a point x (x needs to be positive)

        :param x: A point in (0,infinity)
        :return: The log-density of this distribution at x
        '''
        return log(self.density(x))