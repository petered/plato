from artemis.general.numpy_helpers import get_rng
from artemis.general.should_be_builtins import bad_value
from artemis.ml.tools.neuralnets import initialize_network_params
from plato.interfaces.helpers import get_named_activation_function, batch_normalize
from plato.core import create_shared_variable, symbolic_simple, symbolic
from plato.interfaces.interfaces import IParameterized
import theano.tensor as tt
import numpy as np


@symbolic_simple
class MultiLayerPerceptron(IParameterized):
    """
    A Multi-Layer Perceptron.
    """

    def __init__(self, layers):
        """
        :param weights: A list of initial weight matrices of shapes:
            [(n_in, n_hidden_0), (n_hidden_0, n_hidden_1), ... (n_hidden_N, n_out)]
        :param hidden_activation: A string or list of strings indicating the type of each hidden layer.
            {'sig', 'tanh', 'rect-lin', 'lin', 'softmax'}
        :param output_activation: A string (see above) identifying the activation function for the output layer
        :param normalize_minibatch: True to normalize by mean and standard-deviation over the minibatch.
        :param scale_param: Add a parameter in addition to the biases for rescaling before the nonlinearity.  Only
            really makes sense when normalize_minibatch is True.
        :param use_bias: If False, do not use biases.
        """
        self.layers = layers

    def __call__(self, x):
        for lay in self.layers:
            x = lay(x)
        return x

    @symbolic
    def get_layer_activations(self, x):
        activations = []
        for lay in self.layers:
            x = lay(x)
            activations.append(x)
        return tuple(activations)

    @property
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters]

    @classmethod
    def from_init(cls, w_init, layer_sizes, w_init_dist='normal', rng=None, last_layer_zero=False, **init_args):
        """
        :param w_init: Can be:
            - A scalar, in which case w_init will be interpreted as the standard deviation for the Normally distributed initial weights.
            - A function which accepts the shape of the weight matrix as separate arguments.
        :param layer_sizes: A list of layer sizes, including the input layer
        :param rng: A random number generator or seed to use for drawing weights (only when w_init is a scalar)
        :param last_layer_zero: There is no need for the last layer to have initial weights.  If this is True, the weights of
            the last layer will all be zero.
        :param **init_args: See MultiLayerPerceptron constructor
        """
        assert len(layer_sizes)>1, "A Multi-layer perceptron with only 1 layer don't make no sense.  Remember, input/output layers must be specified."
        if hasattr(w_init, '__call__'):
            assert rng is None, "If w_init is callable, the random number generator (rng) doesn't do anything, and shouldn't be specified."
        else:
            rng = get_rng(rng)
            w_init_mag = w_init
        #     w_init = lambda n_inputs, n_outputs: w_init_mag * rng.randn(n_in, n_out)
        # weights = [w_init(n_in, n_out) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]

            weights = initialize_network_params(layer_sizes=layer_sizes, mag=w_init, base_dist=w_init_dist, include_biases=False, rng=rng)

        if last_layer_zero:
            weights[-1][:] = 0
        return cls.from_weights(weights=weights, **init_args)

    @classmethod
    def from_weights(cls, weights, hidden_activation = 'sig', output_activation = 'sig',
            normalize_minibatch = False, scale_param = False, use_bias = True):
        assert all(w_in.shape[-1] == w_out.shape[-2] for w_in, w_out in zip(weights[:-1], weights[1:]))
        n_layers = len(weights)
        layers = [
            Layer(
                linear_transform=FullyConnectedTransform(
                    w=w,
                    normalize_minibatch=normalize_minibatch if layer_no < (n_layers - 1) else None,
                    scale=scale_param,
                    use_bias=use_bias
                ),
                nonlinearity=nonlinearity
            )
            for w, nonlinearity, layer_no in
            zip(weights, [hidden_activation] * (n_layers - 1) + [output_activation], xrange(n_layers))
            ]
        return MultiLayerPerceptron(layers)


@symbolic_simple
class Layer(IParameterized):
    """
    The composition of a linear transform and a nonlinearity.
    """

    def __init__(self, linear_transform, nonlinearity):
        """
        linear_transform: Can be:
            A callable (e.g. FullyConnectedBridge/ConvolutionalBridge) which does a linear transform on the data.
            A numpy array - in which case it will be used to instantiate a linear transform.
        """
        if isinstance(linear_transform, np.ndarray):
            assert (linear_transform.ndim == 2 and nonlinearity!='maxout') or (linear_transform.ndim == 3 and nonlinearity=='maxout'), \
                'Your weight matrix must be 2-D (or 3-D if you have maxout units)'
            linear_transform = FullyConnectedTransform(w=linear_transform)
        if isinstance(nonlinearity, str):
            nonlinearity = get_named_activation_function(nonlinearity)
        self.linear_transform = linear_transform
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        pre_sig = self.linear_transform(x)
        return self.nonlinearity(pre_sig)

    @property
    def parameters(self):
        return self.linear_transform.parameters


@symbolic_simple
class FullyConnectedTransform(IParameterized):
    """
    An element which multiplies the input by some weight matrix w and adds a bias.
    """

    def __init__(self, w, b = 0, normalize_minibatch = False, scale = False, use_bias = True):
        """
        :param w: Initial weight value.  Can be:
            - A numpy array, in which case a shared variable is instantiated from this data.
            - A symbolic variable that is either a shared variabe or descended from a shared variable.
              This is used when there are shared parameters.
        :param b: Can be:
            - A numpy vector representing the initial bias on the hidden layer, where len(b) = w.shape[1]
            - A scaler, which just initializes the full vector to this value
        :param normalize_minibatch: Set to True to normalize over the minibatch.  This has been shown to cause better optimization
        :param scale: Set to True to include an scale term (per output).  Generally this only makes sense if
            normalize_minibatch is True.
        :param use_bias: Use a bias term?  Generally, the answer is "True", a bias term helps.
        """
        self.w = create_shared_variable(w, name = 'w')
        self.b = create_shared_variable(b, shape = w.shape[1] if w.ndim==2 else (w.shape[0], w.shape[2]) if w.ndim==3 else bad_value(w.shape), name = 'b')
        self.log_scale = create_shared_variable(0 if scale else None, shape = w.shape[1], name = 'log_scale') if scale else None
        self.normalizer = \
            batch_normalize if normalize_minibatch is True else \
            None if normalize_minibatch is False else \
            normalize_minibatch
        self._use_bias = use_bias

    def __call__(self, x):
        current = x.flatten(2).dot(self.w)
        current = self.normalizer(current) if self.normalizer is not None else current
        if self.log_scale is not None:
            current = current * tt.exp(self.log_scale)
        y = (current + self.b) if self._use_bias else current
        return y

    @property
    def parameters(self):
        return [self.w] + ([self.b] if self._use_bias else []) + ([self.log_scale] if self.log_scale is not None else [])


def create_maxout_network(layer_sizes, maxout_widths, w_init, output_activation = 'maxout', rng = None, **other_args):

    rng = get_rng(rng)

    n_expected_maxout_widths = len(layer_sizes)-1 if output_activation=='maxout' else len(layer_sizes)-2
    if isinstance(maxout_widths, (list, tuple)):
        assert len(maxout_widths) == n_expected_maxout_widths
    else:
        maxout_widths = [maxout_widths]*n_expected_maxout_widths

    weights = [w_init*rng.randn(n_maps, n_in, n_out) for n_maps, n_in, n_out in zip(maxout_widths, layer_sizes[:-1], layer_sizes[1:])]
    # Note... we're intentionally starting the zip with maxout widths because we know it may be one element shorter than the layer-sizes
    if output_activation != 'maxout':
        weights.append(w_init*rng.randn(layer_sizes[-2], layer_sizes[-1]))
    return MultiLayerPerceptron.from_weights(weights=weights, hidden_activation='maxout', output_activation=output_activation, **other_args)
