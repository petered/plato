from argmaxlab.spiking_experiments.spike_sampling import get_rng
from plato.interfaces.decorators import symbolic_updater, symbolic_stateless
from plato.interfaces.helpers import get_theano_rng, get_named_activation_function
from plato.tools.cost import mean_squared_error
from plato.tools.online_prediction.online_predictors import ISymbolicPredictor
from plato.tools.optimizers import SimpleGradientDescent
import theano
import theano.tensor as tt

__author__ = 'peter'


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
        return y.dot(self.w_rev + self.b_rev)

    @symbolic_updater
    def train(self, x, target):
        out = self.predict(x)
        target_loss = self.cost_function(out, target)
        forward_updates = self.forward_optimizer(cost = target_loss, parameters = [self.w, self.b])
        noisy_x = x + self.noise*self.rng.normal(size = x.tag.test_value.shape)
        recon = self.backward(self.predict(noisy_x))
        recon_loss = self.cost_function(recon, noisy_x)
        backward_updates = self.backward_optimizer(cost = recon_loss, parameters = [self.w_rev, self.b_rev])
        return forward_updates+backward_updates

    @symbolic_stateless
    def backpropagate_target(self, x, target):
        return x - self.backward(self.predict(x)) + self.backward(target)


class DifferenceTargetMLP(ISymbolicPredictor):

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

        target = xs[-1] - 0.5 * tt.grad(global_loss, xs[-1])

        updates = []
        for i, l in reversed(list(enumerate(self.layers))):
            these_updates = l.train(xs[i], target)
            target = l.backpropagate_target(xs[i], target)
            updates+=these_updates
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
