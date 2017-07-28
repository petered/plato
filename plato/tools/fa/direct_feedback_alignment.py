import numpy as np
from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import izip_equal
from artemis.ml.tools.neuralnets import initialize_weight_matrix
from plato.core import create_shared_variable
from plato.interfaces.helpers import get_named_activation_function, get_named_activation_function_derivative
from plato.tools.mlp.manual_backprop_net import IManualBackpropLayer, ManualBackpropNet
import theano.tensor as tt

class DirectFeedbackAlignmentLayer(IManualBackpropLayer):
    """



    """

    def __init__(self, w, w_back, nonlinearity, b=None, backwards_nonlinearity = 'deriv'):
        self.n_in, self.n_out = w.shape

        self.w = create_shared_variable(w)
        self.b = create_shared_variable(np.zeros(w.shape[1]) if b is None else b)
        if w_back is None:
            self.w_back = None
        else:
            assert w_back.shape[1] == self.n_out
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
        return self.nonlinearity(pre_sig), (x, pre_sig, )

    def backward_pass(self, state, grad, cost):
        # assert cost is None, 'You need to initialize the outer network with pass_loss = False'
        x, pre_sig = state
        if cost is not None:  # Just top layer
            assert self.w_back is None
            grad = this_grad = tt.grad(cost, wrt=pre_sig)
        else:  # Other layers
            this_grad = grad.dot(self.w_back)
        grad_presig = this_grad * self.backwards_nonlinearity(pre_sig)
        return grad, [x.T.dot(grad_presig), grad_presig.mean(axis=0)]

    @classmethod
    def from_init(cls, n_in, n_out, n_final, w_init='xavier', rng=None, **kwargs):
        rng = get_rng(rng)
        w = initialize_weight_matrix(n_in=n_in, n_out=n_out, mag=w_init, rng=rng)
        w_back = None if n_final is None else initialize_weight_matrix(n_in=n_final, n_out=n_out, mag=w_init, rng=rng)
        return cls(w=w, w_back=w_back, **kwargs)


def create_direct_feedback_alignment_net(layer_sizes, nonlinearity, final_nonlinearity, optimizer, loss,
        backwards_nonlinearity='deriv', w_init = 'xavier', rng = None):

    rng = get_rng(rng)
    return ManualBackpropNet(
        layers = [
            DirectFeedbackAlignmentLayer.from_init(
                n_in=n_in,
                n_out=n_out,
                n_final=layer_sizes[-1] if i < len(layer_sizes)-2 else None,
                nonlinearity=nonlinearity if i < len(layer_sizes)-2 else final_nonlinearity,
                backwards_nonlinearity = backwards_nonlinearity,
                w_init = w_init,
                rng=rng
                ) for i, (n_in, n_out) in enumerate(izip_equal(layer_sizes[:-1], layer_sizes[1:]))
            ],
        optimizer=optimizer,
        pass_loss=True,
        loss=loss
        )
