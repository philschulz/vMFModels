import numpy as np
from scipy.special import digamma
from scipy.stats import gamma
from typing import Tuple
import mxnet as mx


class GammaDist(object):
    '''
    Class that implements the gamma distribution with shape-scale parametrisation. The density is
    p(x) = (x**(shape-1) exp(-x/scale))/gamma(shape)*(scale**shape)
    '''

    def __init__(self, shape: float, scale: float):
        '''Constructor

        :param shape: The shape parameter of the distribution
        :param scale: The scale parameter of the distribution
        '''
        self.shape = shape
        self.scale = scale
        self.inv_scale = 1 / scale

    def density(self, x: np.array) -> np.array:
        '''Compute the density a point x (x needs to be positive)

        :param x: A point in (0,infinity)
        :return: The density of this distribution at x
        '''
        return gamma.pdf(x, self.shape, 0, self.scale)

    def log_density(self, x: np.array) -> Tuple[np.array, np.array]:
        '''Compute the log-density at a point x (x needs to be positive)

        :param x: A point in (0,infinity)
        :return: The log-density of this distribution at x
        '''
        density = gamma.logpdf(x, self.shape, 0, self.scale)
        x_gradient = (self.shape - 1) / x - self.inv_scale

        return density, x_gradient


class GammaDistribution(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        shape = in_data[0]
        rate = in_data[1]
        label =  in_data[2]

        density = shape * mx.nd.log(rate) - mx.nd.gammaln(data=shape) + (shape - 1) * mx.nd.log(label) - rate * label

        self.assign(out_data[0], req[0], density)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        shape = in_data[0]
        rate = in_data[1]
        label = in_data[2]

        # TODO: this should be doable with autograd
        shape_grad = mx.nd.log(rate) - mx.nd.array(digamma(shape.asnumpy())) + mx.nd.log(label)
        rate_grad = shape / rate - label
        label_grad = (shape - 1) / label - rate

        self.assign(in_grad[0], req[0], shape_grad)
        self.assign(in_grad[1], req[1], rate_grad)
        self.assign(in_grad[2], req[2], label_grad)


@mx.operator.register("gammaDist")
class GammaDistributionProp(mx.operator.CustomOpProp):

    def list_arguments(self):
        return ["shape", "rate", "label"]

    def list_outputs(self):
        return ["density"]

    def infer_shape(self, in_shape):
        shape_shape = in_shape[0]
        rate_shape = in_shape[1]
        label_shape = in_shape[2]

        output_shape = label_shape

        return [shape_shape, rate_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return GammaDistribution()


# Test code
#
# shape_shape = (100,)
# rate_shape = (100,)
# label_shape = (100,)
# shape = mx.nd.random.gamma(shape=shape_shape)
# rate = mx.nd.random.gamma(shape=rate_shape)
# label = mx.nd.random.gamma(shape=label_shape)
#
# batch = mx.io.DataBatch(data=[shape, rate], label=[label])
#
# shape_var = mx.sym.var("shape")
# rate_var = mx.sym.var("rate")
# label_var = mx.sym.Variable("label")
#
# density = mx.sym.Custom(shape=shape_var, rate=rate_var, label=label_var, op_type="gammaDist")
# loss = mx.sym.MakeLoss(density)
#
# mod = mx.module.Module(loss, data_names=["shape", "rate"], label_names=["label"])
# mod.bind(data_shapes=[("shape", shape_shape), ("rate", rate_shape)], label_shapes=[("label", label_shape)])
# mod.init_params()
# mod.init_optimizer()
#
# mod.forward_backward(batch)
#
# print(mx.nd.exp(mod.get_outputs()[0]))