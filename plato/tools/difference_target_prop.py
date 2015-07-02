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

    def __init__(self, w, b, w_rev, b_rev, input_activation = 'tanh', output_activation = 'tanh', rng = None, noise = 1,
                 optimizer_constructor = lambda: SimpleGradientDescent(0.01), cost_function = mean_squared_error):

        self.noise = noise
        self.rng = get_theano_rng(rng)
        self.w = theano.shared(w, name = 'w')
        self.b = theano.shared(b, name = 'b')
        self.w_rev = theano.shared(w_rev, name = 'w_rev')
        self.b_rev = theano.shared(b_rev, name = 'b_rev')
        self.input_activation = get_named_activation_function(input_activation)
        self.hidden_activation = get_named_activation_function(output_activation)
        self.forward_optimizer = optimizer_constructor()
        self.backward_optimizer = optimizer_constructor()
        self.cost_function = cost_function

    @symbolic_stateless
    def predict(self, x):
        return self.hidden_activation(x.dot(self.w)+self.b)

    @symbolic_stateless
    def backward(self, y):
        return self.input_activation(y.dot(self.w_rev) + self.b_rev)

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


class DifferenceTargetLayer(ISymbolicPredictor):

    def __init__(self, w, b, w_rev, b_rev, input_activation = 'tanh', output_activation = 'tanh', rng = None, noise = 1,
                 optimizer_constructor = lambda: SimpleGradientDescent(0.01), cost_function = mean_squared_error):

        self.noise = noise
        self.rng = get_theano_rng(rng)
        self.w = theano.shared(w, name = 'w')
        self.b = theano.shared(b, name = 'b')
        self.w_rev = theano.shared(w_rev, name = 'w_rev')
        self.b_rev = theano.shared(b_rev, name = 'b_rev')
        self.input_activation = get_named_activation_function(input_activation)
        self.hidden_activation = get_named_activation_function(output_activation)
        self.forward_optimizer = optimizer_constructor()
        self.backward_optimizer = optimizer_constructor()
        self.cost_function = cost_function

    @symbolic_stateless
    def predict(self, x):
        return self.hidden_activation(x.dot(self.w)+self.b)

    @symbolic_stateless
    def backward(self, y):
        return self.input_activation(y.dot(self.w_rev) + self.b_rev)

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


class ReversedDifferenceTargetLayer(DifferenceTargetLayer):
    """
    This is an experimental modification where we switch the order of the linear/nonlinear
    operations.  That is, instead of the usual
        f(x) = activation(x.dot(w))
    We do
        f(x) = activation(x).dot(w)

    We just want to see if this works.
    """

    @symbolic_stateless
    def predict(self, x):
        pre_sigmoid = x.dot(self.w)+self.b
        output = self.hidden_activation(pre_sigmoid)
        output.pre_sigmoid = pre_sigmoid
        return output

    # @symbolic_stateless
    # def backward(self, y):
    #     return self.input_activation(y.dot(self.w_rev) + self.b_rev)

    def backpropagate_target(self, x, target):

        back_output_pre_sigmoid = self.predict(x).dot(self.w_rev) + self.b_rev
        back_target_pre_sigmoid = target.dot(self.w_rev) + self.b_rev

        return self.input_activation(x.pre_sigmoid - back_output_pre_sigmoid + back_target_pre_sigmoid)

        # return x - self.backward(self.predict(x)) + self.backward(target)



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

    def __init__(self, layers, optimizer = SimpleGradientDescent(0.01), cost_function = mean_squared_error, output_cost_function = mean_squared_error):
        """
        (Use from_initializer constructor for more direct parametrization)
        :param layers: A list of DifferenceTargetLayers
        :param optimizer: An IGradientOptimizer object
        :param cost_function: A cost function of the form cost=cost_fcn(actual, target)
        """
        self.layers = layers
        self.optimizer = optimizer
        self.cost_function = cost_function
        self.output_cost_function = output_cost_function

    @classmethod
    def from_initializer(cls, input_size, output_size, hidden_sizes = [], w_init_mag = 0.01, rng = None,
            input_activation = 'linear', hidden_activation = 'tanh', output_activation = 'softmax',
            layer_constructor = DifferenceTargetLayer, output_cost_function = mean_squared_error, **kwargs):
        """
        Create a Difference Target MLP.

        :param input_size: Size of the input
        :param output_size: Size of the output
        :param hidden_sizes: List of integers indicating the sizes of hidden layers
        :param w_init_mag: Magnitude of initial weights
        :param rng: Random number generator/seed (for weight initialization and noise injection)
        :param input_activation: Activation function for the input layer
        :param hidden_activation: Activation function for the hidden layers
        :param output_activation: Activation function for the output layer.
        :param layer_constructor: Constructor for the layer.
        :param kwargs: See __init__ constructor
        :return: A DifferenceTargetMLP
        """

        rng = get_rng(rng)
        all_layer_sizes = [input_size]+hidden_sizes+[output_size]
        all_layer_activations = [input_activation] + [hidden_activation]*len(hidden_sizes) + [output_activation]

        return cls([
            layer_constructor(
                w = w_init_mag*rng.randn(n_in, n_out),
                b = w_init_mag*rng.randn(n_out),
                w_rev = w_init_mag*rng.randn(n_out, n_in),
                b_rev = w_init_mag*rng.randn(n_in),
                input_activation = act_in,
                output_activation = act_out,
                rng = rng,
                **kwargs
                )
            for n_in, n_out, act_in, act_out in zip(all_layer_sizes[:-1], all_layer_sizes[1:],
                all_layer_activations[:-1], all_layer_activations[1:])
            ],
            output_cost_function = output_cost_function
        )

    @symbolic_updater
    def train(self, x, target):

        xs = [x]
        for l in self.layers:
            xs.append(l.predict(xs[-1]))

        global_loss = self.output_cost_function(xs[-1], target)

        # Note: This 0.5 thing is not always the way it should be done.
        # 0.5 is chosen because this makes it equivalent to the regular loss with MSE,
        # but it's unknown whether this is the best way to go.
        target = xs[-1] - 0.5 * tt.grad(global_loss, xs[-1])

        updates = []
        for i, l in reversed(list(enumerate(self.layers))):
            local_updates = l.train(xs[i], target)
            updates+=local_updates
            if i>0:  # No need to go to initial layer
                target = l.backpropagate_target(xs[i], target)

        return updates

    @symbolic_stateless
    def predict(self, x):
        for l in self.layers:
            x = l.predict(x)
        return x