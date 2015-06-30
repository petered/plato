from argmaxlab.spiking_experiments.spike_sampling import get_rng
from plato.interfaces.decorators import symbolic_updater, symbolic_stateless
from plato.interfaces.helpers import get_theano_rng, get_named_activation_function
from plato.tools.cost import mean_squared_error
from plato.tools.online_prediction.online_predictors import ISymbolicPredictor
from plato.tools.optimizers import SimpleGradientDescent
import theano
import theano.tensor as tt

__author__ = 'peter'


class DifferenceTargetMLP(ISymbolicPredictor):
    """
    An MLP doing Difference Target Propagation

    See:
        Dong-Hyun Lee, Saizheng Zhang, Asja Fischer, Antoine Biard1, Yoshua Bengio1
        Difference Target Propagation
        http://arxiv.org/pdf/1412.7525v3.pdf

    Notes:
    - See demo_compare_optimizers for a comparison between this and a regular MLP
    - Currently uses a eta=0.5 for the final layer.  Not sure if this is always a good
      choice - see paper.
    """

    def __init__(self, layers, optimizer = SimpleGradientDescent(0.01), cost_function = mean_squared_error):
        self.layers = layers
        self.optimizer = optimizer
        self.cost_function = cost_function

    @symbolic_updater
    def train(self, x, target):

        xs = [x]
        for l in self.layers:
            xs.append(l.predict(xs[-1]))

        global_loss = self.cost_function(xs[-1], target)

        # Note: This 0.5 thing is not always the way it should be done.
        # 0.5 is chosen because this makes it equivalent to the regular loss with MSE,
        # but it's unknown whether this is the best way to go.
        target = xs[-1] - 0.5 * tt.grad(global_loss, xs[-1])

        updates = []
        for i, l in reversed(list(enumerate(self.layers))):
            local_updates = l.train(xs[i], target)
            updates+=local_updates
            target = l.backpropagate_target(xs[i], target)

        return updates

    @symbolic_stateless
    def predict(self, x):
        for l in self.layers:
            x = l.predict(x)
        return x

    @classmethod
    def from_initializer(cls, input_size, output_size, hidden_sizes = [], w_init_mag = 0.01, rng = None, **kwargs):

        rng = get_rng(rng)
        all_layer_sizes = [input_size]+hidden_sizes+[output_size]

        return cls([
            DifferenceTargetLayer(
                w = w_init_mag*rng.randn(n_in, n_out),
                b = w_init_mag*rng.randn(n_out),
                w_rev = w_init_mag*rng.randn(n_out, n_in),
                b_rev = w_init_mag*rng.randn(n_in),
                rng = rng,
                **kwargs
                )
            for n_in, n_out in zip(all_layer_sizes[:-1], all_layer_sizes[1:])
            ]
        )


class DifferenceTargetLayer(ISymbolicPredictor):

    def __init__(self, w, b, w_rev, b_rev, activation = 'tanh', rng = None, noise = 1,
                 optimizer_constructor = lambda: SimpleGradientDescent(0.01), cost_function = mean_squared_error):

        self.noise = noise
        self.rng = get_theano_rng(rng)
        self.w = theano.shared(w, name = 'w')
        self.b = theano.shared(b, name = 'b')
        self.w_rev = theano.shared(w_rev, name = 'w_rev')
        self.b_rev = theano.shared(b_rev, name = 'b_rev')
        self.activation = get_named_activation_function(activation)
        self.forward_optimizer = optimizer_constructor()
        self.backward_optimizer = optimizer_constructor()
        self.cost_function = cost_function

    @symbolic_stateless
    def predict(self, x):
        return self.activation(x.dot(self.w)+self.b)

    @symbolic_stateless
    def backward(self, y):
        return y.dot(self.w_rev) + self.b_rev

    @symbolic_updater
    def train(self, x, target):
        out = self.predict(x)
        forward_updates = self.forward_optimizer(
            cost = self.cost_function(out, target),
            parameters = [self.w, self.b], constants = [target]
            )  # The "constants" (above) is really important - otherwise it can just try to change the target (which is a function of the weights too).
        noisy_x = x + self.noise*self.rng.normal(size = x.tag.test_value.shape)
        recon = self.backward(self.predict(noisy_x))
        backward_updates = self.backward_optimizer(
            cost = self.cost_function(recon, noisy_x),
            parameters = [self.w_rev, self.b_rev])
        return forward_updates+backward_updates

    @symbolic_stateless
    def backpropagate_target(self, x, target):
        return x - self.backward(self.predict(x)) + self.backward(target)
