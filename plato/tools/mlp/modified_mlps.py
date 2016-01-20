from general.numpy_helpers import get_rng
from general.should_be_builtins import bad_value
from plato.core import symbolic_single_output_updater
from plato.interfaces.helpers import create_shared_variable, batch_normalize, get_named_activation_function
from plato.interfaces.interfaces import IParameterized
from plato.tools.misc.tdb_plotting import tdbplot
import theano
import theano.tensor as tt
import numpy as np

__author__ = 'peter'


@symbolic_single_output_updater
class StatefulMultiLayerPerceptron(object):
    """
    A Multi-Layer Perceptron.  This version is deprecated.  Use plato.tools.mlp.MultiLayerPerceptron, which has a
    new constructor format.
    """

    def __init__(self, weights, hidden_activation = 'sig', output_activation = 'sig',
            normalize_minibatch = False, scale_param = False, use_bias = True):
        """
        :param layer_sizes: A list indicating the sizes of each layer, including the input layer.
        :param hidden_activation: A string or list of strings indicating the type of each hidden layer.
            {'sig', 'tanh', 'rect-lin', 'lin', 'softmax'}
        :param output_activation: A string (see above) identifying the activation function for the output layer
        :param w_init: A function which, given input dims, output dims, return
        """

        assert all(w_in.shape[-1]==w_out.shape[-2] for w_in, w_out in zip(weights[:-1], weights[1:]))
        n_layers = len(weights)

        self.layers = [
            StatefulLayer(
                linear_transform = StatefulFullyConnectedTransform(
                    w = w,
                    normalize_minibatch=normalize_minibatch if layer_no<(n_layers-1) else None,  # It makes no sense to do this to the last layer, because we'll never be able to hit the target unless it too is normalized.
                    scale = scale_param,
                    use_bias = use_bias
                    ),
                nonlinearity = nonlinearity
                )
            for w, nonlinearity, layer_no in zip(weights, [hidden_activation] * (n_layers-1) + [output_activation], xrange(n_layers))
            ]

    def __call__(self, x):
        outputs, updates = theano.scan(
            fn = self.process_sample,
            sequences = [x[:, None, :]],
            )
        return outputs[:, 0, :], updates

    def process_sample(self, x):
        assert x.tag.test_value.shape[0] == 1, "We expect x to have a minibatch of size 1."
        all_updates = []
        for lay in self.layers:
            x, updates = lay.to_format(symbolic_single_output_updater)(x)
            all_updates+=updates
        return x, all_updates


    @property
    def parameters(self):
        return sum([l.parameters for l in self.layers], [])

    @staticmethod
    def from_init(w_init_mag, layer_sizes, rng=None, last_layer_zero=False, **init_args):
        rng = get_rng(rng)
        weights = [w_init_mag*rng.randn(n_in, n_out) for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:])]
        if last_layer_zero:
            weights[-1][:] = 0
        return StatefulMultiLayerPerceptron(weights=weights, **init_args)


@symbolic_single_output_updater
class StatefulLayer(IParameterized):
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
            linear_transform = StatefulFullyConnectedTransform(w=linear_transform)
        if isinstance(nonlinearity, str):
            nonlinearity = get_named_activation_function(nonlinearity)
        self.linear_transform = linear_transform
        self.nonlinearity = nonlinearity

    def __call__(self, x):
        pre_sig, state_updates = self.linear_transform(x)
        return self.nonlinearity(pre_sig), state_updates

    @property
    def parameters(self):
        return self.linear_transform.parameters


@symbolic_single_output_updater
class StatefulFullyConnectedTransform(IParameterized):
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

        # tdbplot(self.w, 'w-%s'%self)

    def __call__(self, x):
        current = x.flatten(2).dot(self.w)

        if self.normalizer is not None:
            current, normalization_updates = self.normalizer.to_format(symbolic_single_output_updater)(current)
        else:
            normalization_updates = []

        if self.log_scale is not None:
            current = current * tt.exp(self.log_scale)
        y = (current + self.b) if self._use_bias else current
        return y, normalization_updates

    @property
    def parameters(self):
        return [self.w] + ([self.b] if self._use_bias else []) + ([self.log_scale] if self.log_scale is not None else [])
