from plato.interfaces.decorators import symbolic_simple
from plato.interfaces.helpers import get_named_activation_function, create_shared_variable
from plato.interfaces.interfaces import IParameterized
import theano.tensor as tt
import numpy as np


@symbolic_simple
class MultiLayerPerceptron(IParameterized):
    """
    A Multi-Layer Perceptron
    """

    def __init__(self, layer_sizes, input_size, hidden_activation = 'sig', output_activation = 'sig',
            normalize_minibatch = False, scale_param = False, w_init = 0.1, use_bias = True):
        """
        :param layer_sizes: A list indicating the sizes of each layer.
        :param input_size: An integer indicating the size of the input layer
        :param hidden_activation: A string or list of strings indicating the type of each hidden layer.
            {'sig', 'tanh', 'rect-lin', 'lin', 'softmax'}
        :param output_activation: A string (see above) identifying the activation function for the output layer
        :param w_init: A function which, given input dims, output dims, return
        """
        if isinstance(w_init, (int, float)):
            val = w_init
            w_init = lambda n_in, n_out: val*np.random.randn(n_in, n_out)

        self.layers = [
            Layer(
                linear_transform = FullyConnectedTransform(
                    w = w_init(pre_size, post_size),
                    normalize_minibatch=normalize_minibatch,
                    scale = scale_param,
                    use_bias = use_bias
                    ),
                nonlinearity = nonlinearity
                )
            for pre_size, post_size, nonlinearity in zip(
                [input_size]+layer_sizes[:-1],
                layer_sizes,
                [hidden_activation] * (len(layer_sizes)-1) + [output_activation]
            )]

    def __call__(self, x):
        for lay in self.layers:
            x = lay(x)
        return x

    @property
    def parameters(self):
         return sum([l.parameters for l in self.layers], [])


def normal_w_init(mag, seed = None):
    rng = np.random.RandomState(seed)
    return lambda n_in, n_out: mag * rng.randn(n_in, n_out)


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
            assert linear_transform.ndim == 2, 'This just works for 2-d arrays right now.'
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
        self.b = create_shared_variable(b, shape = w.shape[1], name = 'b')
        self.log_scale = create_shared_variable(0 if scale else None, shape = w.shape[1], name = 'log_scale') if scale else None
        self._normalize_minibatch = normalize_minibatch
        self._use_bias = use_bias

    def __call__(self, x):
        current = x.flatten(2).dot(self.w)
        if self._normalize_minibatch:
            current = (current - current.mean(axis = 0, keepdims = True)) / (current.std(axis = 0, keepdims = True) + 1e-9)
        if self.log_scale is not None:
            current = current * tt.exp(self.log_scale)
        y = (current + self.b) if self._use_bias else current
        return y

    @property
    def parameters(self):
        return [self.w] + ([self.b] if self._use_bias else []) + ([self.log_scale] if self.log_scale is not None else [])


@symbolic_simple
class ConvolutionalTransform(IParameterized):

    def __init__(self, w, b=0, stride = (1, 1)):
        self.w = create_shared_variable(w, name = 'w')
        self.b = create_shared_variable(b, name = 'b')
        self._stride = stride

    def __call__(self, x):
        y = tt.nnet.conv2d(x, self._w, border_mode='valid', subsample = self._stride) + self._b.dimshuffle('x', 0, 'x', 'x')
        return y

    @property
    def parameters(self):
        return [self.w, self.b]
