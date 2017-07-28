import numpy as np
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import izip_equal
from artemis.ml.tools.neuralnets import initialize_weight_matrix
from plato.core import create_shared_variable
from plato.interfaces.helpers import get_named_activation_function, get_named_activation_function_derivative
from plato.tools.mlp.manual_backprop_net import IManualBackpropLayer, ManualBackpropNet
from theano import tensor as tt


class FeedbackAlignmentLayer(IManualBackpropLayer):

    def __init__(self, w, w_back, nonlinearity, b=None, backwards_nonlinearity = 'deriv'):
        self.n_in, self.n_out = w.shape

        assert w_back.shape == (self.n_out, self.n_in)
        self.w = create_shared_variable(w)
        self.b = create_shared_variable(np.zeros(w.shape[1]) if b is None else b)
        self.w_back = create_shared_variable(w_back)

        self.nonlinearity = get_named_activation_function(nonlinearity) if isinstance(nonlinearity, str) else nonlinearity
        self.backwards_nonlinearity = \
            get_named_activation_function_derivative(nonlinearity) if backwards_nonlinearity=='deriv' else \
            get_named_activation_function(backwards_nonlinearity) if isinstance(backwards_nonlinearity, basestring) else \
            backwards_nonlinearity

    @property
    def parameters(self):
        return [self.w, self.b]

    def forward_pass_and_state(self, x):
        pre_sig = x.dot(self.w)
        out = self.nonlinearity(pre_sig)
        return out, (x, pre_sig, out)

    def backward_pass(self, state, grad, cost):
        x, pre_sig, out = state
        if grad is None:
            grad_presig = tt.grad(cost, wrt = pre_sig)
        else:
            # return self.backward_pass(state=state, grad=grad, cost=None)
            assert cost is None and grad is not None
            grad_presig = grad * self.backwards_nonlinearity(pre_sig)
        return grad_presig.dot(self.w_back), [x.T.dot(grad_presig), grad_presig.mean(axis=0)]

    @classmethod
    def from_init(cls, n_in, n_out, w_init='xavier', rng=None, **kwargs):
        rng = get_rng(rng)
        w = initialize_weight_matrix(n_in=n_in, n_out=n_out, mag=w_init, rng=rng)
        w_back = initialize_weight_matrix(n_in=n_out, n_out=n_in, mag=w_init, rng=rng)
        return cls(w=w, w_back=w_back, **kwargs)


def create_feedback_alignment_net(layer_sizes, nonlinearity, final_nonlinearity, optimizer, loss, backwards_nonlinearity='deriv',
        w_init = 'xavier', rng = None):

    rng = get_rng(rng)
    return ManualBackpropNet(
        layers = [
            FeedbackAlignmentLayer.from_init(
                n_in=n_in,
                n_out=n_out,
                nonlinearity=nonlinearity if i < len(layer_sizes)-2 else final_nonlinearity,
                backwards_nonlinearity = backwards_nonlinearity,
                w_init = w_init,
                rng=rng
                ) for i, (n_in, n_out) in enumerate(izip_equal(layer_sizes[:-1], layer_sizes[1:]))
            ],
        optimizer=optimizer,
        loss=loss
        )