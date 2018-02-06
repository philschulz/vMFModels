import mxnet as mx
import numpy as np
from scipy.special import iv as bessel


class Bessel(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        order = in_data[0]
        data = in_data[1].asnumpy()

        result = mx.nd.array(bessel(order, data))

        self.assign(out_data[0], req[0], result)


    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        order = in_data[0]
        data = in_data[1].asnumpy()

        bessel_grad = mx.nd.array((bessel(order - 2, data) - bessel(order + 1, data)) / 2)
        grad = out_grad[0] * bessel_grad

        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("bessel")
class BesselOpProp(mx.operator.CustomOpProp):

    def list_arguments(self):
        return ["dim", 'data']

    def list_outputs(self):
        return ["output"]

    def infer_shape(self, in_shape):
        out_shape = in_shape[0]

        return [in_shape], [out_shape], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return Bessel()


class LogBessel(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        order = in_data[0]
        data = in_data[1].asnumpy()

        result = mx.nd.array(np.log(bessel(order, data)))

        self.assign(out_data[0], req[0], result)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        order = in_data[0]
        data = in_data[1]

        log_bessel_grad = mx.nd.array((bessel(order - 2, data) - bessel(order + 1, data)) / 2 * bessel(order, data))
        grad = out_grad * log_bessel_grad

        self.assign(in_grad[0], req[0], grad)


@mx.operator.register("log_bessel")
class LogBesselOpProp(mx.operator.CustomOpProp):

    def list_arguments(self):
        return ["order", "data"]

    def list_outputs(self):
        return ["output"]

    def infer_shape(self, in_shape):
        out_shape = in_shape[0]

        return [out_shape], [out_shape], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return LogBessel()


def vmf_normaliser(dim, kappa):

    log_bessel = mx.sym.Custom(dim=dim, data=kappa, op_type="log_bessel")
    return mx.sym.log(kappa) * (dim / 2 - 1) - log_bessel - mx.sym.log(2 * np.pi) * (dim / 2)



class VMFDistribution(mx.operator.CustomOp):

    def log_bessel_gradient(self, dim: int, kappa: mx.nd.array) -> mx.nd.array:
        return (bessel(dim / 2 - 2, kappa) - bessel(dim / 2, kappa)) / 2 * bessel(dim / 2 - 1, kappa)

    def forward(self, is_train, req, in_data, out_data, aux):
        mu = in_data[0]
        dim = mu.shape[-1]
        kappa = mx.nd.expand_dims(data=in_data[1], axis=kappa.ndim)
        np_kappa = kappa.asnumpy()
        label = in_data[2]

        normaliser = mx.nd.array(
            (np.log(np_kappa) * (dim / 2 - 1) - np.log(bessel(dim / 2 - 1, np_kappa))) - (dim / 2) * np.log(2 * np.pi))
        energy = mx.nd.batch_dot(label, mu, transpose_b=True)
        density = mx.nd.broadcast_add(lhs=normaliser, rhs=mx.nd.broadcast_mul(lhs=energy, rhs=kappa))

        self.assign(out_data[0], req[0], density)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        mu = in_data[0]
        kappa = in_data[1]
        kappa = mx.nd.expand_dims(data=kappa, axis=kappa.ndim)
        np_kappa = kappa.asnumpy()
        label = in_data[2]
        dim = label.shape[-1]

        density = out_data[0]
        posterior = density / mx.nd.sum(data=density, axis=density.ndim-1, keepdims=True)

        expected_label_sum = mx.nd.batch_dot(lhs=posterior, rhs=label, transpose_a=True)
        natural_param = mx.nd.broadcast_mul(lhs=kappa, rhs=expected_label_sum)
        norm = mx.nd.sqrt(mx.sum(data=mx.nd.square(data=natural_param), axis=natural_param.ndim-1, keep_dims=True))
        mu_grad = natural_param / norm

        energy = mx.nd.batch_dot(mu, label, transpose_b=True, name="backward_compute_vmf_energy")
        kappa_grad = (dim / 2 - 1) / kappa - mx.nd.array(self.log_bessel_gradient(dim, np_kappa))
        kappa_grad = mx.nd.broadcast_add(lhs=energy, rhs=kappa_grad)
        kappa_grad = mx.nd.sum(data=kappa_grad, axis=kappa_grad.ndim - 1, keepdims=False)

        self.assign(in_grad[0], req[0], mu_grad)
        self.assign(in_grad[1], req[1], kappa_grad)


@mx.operator.register("vmfDist")
class VMFDistributionProp(mx.operator.CustomOpProp):

    def list_arguments(self):
        return ['mu', 'kappa', 'label']

    def list_outputs(self):
        return ['density']

    def infer_shape(self, in_shape):
        mu_shape = in_shape[0]
        kappa_shape = in_shape[1]
        label_shape = in_shape[2]

        output_shape = (mu_shape[0], label_shape[1], mu_shape[1])

        return [mu_shape, kappa_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, in_shapes, in_dtypes):
        return VMFDistribution()

#Test code

source_shape = (2, 10)
target_shape = (2, 8)
source = mx.nd.array(np.random.randint(low=0, high=10, size=source_shape))
target = mx.nd.array(np.random.randint(low=0, high=10, size=target_shape))

batch = mx.io.DataBatch(data=[target], label=[source])

s_sent = mx.sym.var("source")
t_sent = mx.sym.var("target")

source_embed = mx.sym.Embedding(data=s_sent, input_dim=11, output_dim=50)
target_embed = mx.sym.Embedding(data=t_sent, input_dim=11, output_dim=50)
kappa = mx.sym.ones_like(t_sent)

density = mx.sym.Custom(mu=target_embed, kappa=kappa, label=source_embed, op_type="vmfDist")
loss = mx.sym.MakeLoss(data=density)

mod = mx.module.Module(loss, data_names=["target"], label_names=["source"])
mod.bind(data_shapes=[("target", target_shape)], label_shapes=[("source", source_shape)])
mod.init_params()
mod.init_optimizer()

mod.forward_backward(batch)
print(mod.get_outputs())
