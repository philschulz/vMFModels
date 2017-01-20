from random import Random
from math import log, floor

class UnivariateSliceSampler(object):
    '''A univariate slice sampler that is based on the stepping out and shrinkage procedures. At the moment it only
    supports distributions defined on the positive reals.
    '''

    max_steps = 10
    step_size = 1

    def __init__(self, prior_distribution, likelihood = None):
        '''Constructor

        :param prior_distribution: A prior distribution that comes with a log-density function
        :param likelihood: A likelihood function that depends on the value to be sampled
        '''
        self.state = None
        self.random_generator = Random()
        self.prior_dist = prior_distribution
        self.likelihood = likelihood

    def set_likelihood(self, likelihood):
        '''Assigns a new likelihood function

        :param likelihood: A likelihood function that depends on the value to be sampled
        '''
        self.likelihood = likelihood

    def sample(self, starting_point, iterations, burn_in = 0):
        '''Take a number of samples

        :param starting_point: The initial state in the Markov chain
        :param iterations: The number of iterations for which to sample
        :param burn_in: The burn-in period (samples taken in this period are discarded)
        :return: A list of samples and the sample average
        '''
        self.state = starting_point
        samples = list()

        for iter in range(iterations):
            # TODO make sure everything is on log scale
            likelihood = self.likelihood(self.state)
            prior_density = self.prior_dist.log_density(self.state)
            random = log(self.random_generator.random())
            threshold = likelihood + prior_density + random

            # stepping out procedure
            left_initial_step = self.random_generator.random()*self.step_size
            right_initial_step = self.step_size - left_initial_step
            left_edge = (self.state - left_initial_step) if (self.state - left_initial_step) > 0 else 0
            right_edge = self.state + right_initial_step

            left_steps = floor(self.random_generator.random()*self.max_steps)
            right_steps = self.max_steps - 1 - left_steps

            # step out to the left
            for _ in range(left_steps):
                left_edge -= self.step_size
                if left_edge <= 0:
                    left_edge = 0
                    break
                elif self.likelihood(left_edge) + self.prior_dist.log_density(left_edge) < threshold:
                    break

            # step out to the right
            for _ in range(right_steps):
                right_edge += self.step_size
                if self.likelihood(right_edge) + self.prior_dist.log_density(right_edge):
                    break

            # shrinking procedure
            miss = True
            new_value = 0
            while miss:
                new_value = left_edge + self.random_generator.random() * (right_edge - left_edge)
                score = self.likelihood(new_value) + self.prior_dist.log_density(new_value)
                miss = score < threshold
                if miss:
                    if new_value < self.state:
                        left_edge = new_value
                    else:
                        right_edge = new_value

            if iter >= burn_in:
                samples.append(new_value)
            self.state = new_value

        return samples, sum(samples)/len(samples)