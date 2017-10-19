from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as tt
from artemis.general.numpy_helpers import get_rng
from plato.core import symbolic, create_shared_variable
from plato.interfaces.helpers import get_named_activation_function, get_theano_rng
from plato.interfaces.interfaces import IParameterized
from plato.tools.common.online_predictors import FeedForwardModule
from plato.tools.convnet.conv_specifiers import ConvInitSpec, ConvolverSpec, PoolerSpec, NonlinearitySpec, DropoutSpec, \
    FullyConnectedSpec, ConvNetSpec
from theano.tensor.signal.pool import pool_2d
__author__ = 'peter'
import logging

LOGGER = logging.getLogger('plato')

@symbolic
class ConvLayer(FeedForwardModule):

    def __init__(self, w, b, force_shared_parameters = True, border_mode = 'valid', filter_flip = True):
        """
        w is the kernel, an ndarray of shape (n_output_maps, n_input_maps, w_size_y, w_size_x)
        b is the bias, an ndarray of shape (n_output_maps, ).  Can also be "False" meaning, don't use biases

        force_shared_parameters: Set to true if you want to make the parameters shared variables.  If False, the
            parameters will be be constants (which allows for certain optimizations)
        :param border_mode: {'valid', 'full', 'half', int, (int1, int2)}.  Affects
            default is 'valid'.  See theano.tensor.nnet.conv2d docstring for details.
        """
        self.w = create_shared_variable(w) if force_shared_parameters else tt.constant(w)
        self.b = False if b is False else create_shared_variable(b) if force_shared_parameters else tt.constant(b)
        self.border_mode = border_mode
        self.filter_flip = filter_flip

    def __call__(self, x):
        """
        param x: A (n_samples, n_input_maps, size_y, size_x) image/feature tensor
        return: A (n_samples, n_output_maps, size_y-w_size_y+1, size_x-w_size_x+1) tensor
        """
        result = tt.nnet.conv2d(input=x, filters=self.w, border_mode=self.border_mode, filter_flip=self.filter_flip) + (self.b[:, None, None] if self.b is not False else 0)
        return result

    @property
    def parameters(self):
        return [self.w, self.b] if self.b is not False else [self.w]
    
    def to_spec(self):
        return ConvolverSpec(self.w.get_value(), self.b.get_value() if self.b is not False else False, self.border_mode)

    @classmethod
    def from_spec(cls, spec):
        return ConvLayer(
            w=spec.w,
            b=spec.b,
            border_mode= {'full': 0, 'same': 'half', 'valid': 0}[spec.mode] if spec.mode in ('full', 'same', 'valid') else spec.mode,
            filter_flip=False
            )


@symbolic
class Nonlinearity(FeedForwardModule):

    def __init__(self, activation):
        """
        activation:  a name for the activation function. {'relu', 'sig', 'tanh', ...}
        """
        self._activation_name = activation
        self.activation = get_named_activation_function(activation) if isinstance(activation, basestring) else activation

    def __call__(self, x):
        return self.activation(x)

    def to_spec(self):
        assert isinstance(self._activation_name, basestring), "Can't identify activation fcn"
        return NonlinearitySpec(self._activation_name)

    @classmethod
    def from_spec(cls, spec):
        return Nonlinearity(spec.func)


@symbolic
class CrossConvLayer(object):

    def __init__(self, ):

    def __call__(self, (x1, x2)):
        """
        (x1, x2) are each n_samples, n_maps
        :return:
        """
        map = tt.nnet.conv2d(input=x1, filters=x1, )

@symbolic
class Pooler(FeedForwardModule):

    def __init__(self, region, stride = None, mode = 'max'):
        """
        :param region: Size of the pooling region e.g. (2, 2)
        :param stride: Size of the stride e.g. (2, 2) (defaults to match pooling region size for no overlap)
        """
        if isinstance(region, int):
            region = region, region
        if isinstance(stride, int):
            stride = stride, stride
        assert len(region) == 2, 'Region must consist of two integers.  Got: %s' % (region, )
        if stride is None:
            stride = region
        assert len(region) == 2, 'Stride must consist of two integers.  Got: %s' % (region, )
        self.region = region
        self.stride = stride
        self.mode = mode

    def __call__(self, x):
        """
        :param x: An (n_samples, n_maps, size_y, size_x) tensor
        :return: An (n_sample, n_maps, size_y/ds[0], size_x/ds[1]) tensor
        """
        return pool_2d(x, ds = self.region, st = self.stride, mode = self.mode, ignore_border=True)

    def to_spec(self):
        return PoolerSpec(region = self.region, stride=self.stride, mode=self.mode)

    @classmethod
    def from_spec(cls, spec):
        return Pooler(region=spec.region, stride=spec.stride, mode=spec.mode)

@symbolic
class DropoutLayer(FeedForwardModule):

    def __init__(self, dropout_rate, rng = None, shape=None):
        """
        :param dropout_rate: The fraction of units to dropout (0, 1)
        :param rng: Random number generator
        :param shape: Optionally, the shape.  If not

        Returns
        -------

        """
        self.dropout_rate = dropout_rate
        self.rng = get_theano_rng(rng)
        self.shape = shape

    def __call__(self, x):
        dropped_units = self.rng.binomial(n=1, p=self.dropout_rate, size=x.shape if self.shape is None else self.shape, dtype = theano.config.floatX)
        return tt.switch(dropped_units, tt.constant(0, dtype=theano.config.floatX), x)

    def test_call(self, x):
        return x / tt.constant(1 - self.dropout_rate, dtype = theano.config.floatX)

    def to_spec(self):
        return DropoutSpec(self.dropout_rate)

    @classmethod
    def from_spec(cls, spec):
        return cls(spec.dropout_rate, rng=rng)

@symbolic
class FullyConnectedLayer(FeedForwardModule):

    def __init__(self, w, b=None):
        if b is None:
            b = np.zeros(w.shape[1])
        self.w = create_shared_variable(w)
        self.b = create_shared_variable(b)

    def __call__(self, x):
        """
        param x: A (n_samples, n_input_maps, size_y, size_x) image/feature tensor
        return: A (n_samples, n_output_maps, size_y-w_size_y+1, size_x-w_size_x+1) tensor
        """
        result = x.flatten(2).dot(self.w) + self.b
        return result

    @property
    def parameters(self):
        return [self.w, self.b] if self.b is not False else [self.w]

    def to_spec(self):
        return FullyConnectedSpec(w=self.w.get_value(), b=self.b.get_value() if self.b is not False else False)

    @classmethod
    def from_spec(cls, spec):
        return FullyConnectedLayer(w=spec.w, b=spec.b)


@symbolic
class ConvNet(IParameterized):

    def __init__(self, layers):
        """
        :param layers: Either:
            A list of layers or
            An OrderedDict<layer_name: layer>
        """
        self.n_layers = len(layers)
        if isinstance(layers, (list, tuple)):
            layers = OrderedDict(enumerate(layers))
        else:
            assert isinstance(layers, OrderedDict), "Layers must be presented as a list, tuple, or OrderedDict"
        self.layers = layers

    def __call__(self, inp):
        """
        :param inp: An (n_samples, n_colours, size_y, size_x) input image
        :return: An (n_samples, n_feature_maps, map_size_y, map_size_x) feature representation.
        """
        return self.get_named_layer_activations(inp).values()[-1]

    @symbolic
    def test_call(self, inp):
        return self.get_named_layer_activations(inp, test_call=True).values()[-1]

    @symbolic
    def get_named_layer_activations(self, x, include_input = False, test_call=False):
        """
        :param x: A (n_samples, n_colours, size_y, size_x) input image
        :param test_call: True if you want to call it on a test set... this may affect things like dropout.
        :returns: An OrderedDict<layer_name/index, activation>
            If you instantiated the convnet with an OrderedDict, the keys will correspond to the keys for the layers.
            Otherwise, they will correspond to the index which identifies the order of the layer.
        """
        named_activations = OrderedDict([('input', x)]) if include_input else OrderedDict()
        for name, layer in self.layers.iteritems():
            x = layer.test_call(x) if test_call else layer.train_call(x)
            named_activations[name] = x
        return named_activations

    @staticmethod
    def from_init(specifiers, input_shape=None, w_init=0.01, force_shared_parameters = True, rng=None):
        """
        Convenient initialization function.
        :param specifiers: List/OrderedDict of layer speciefier objects (see conv_specifiers.py)
        :param input_shape: Input shape (n_maps, n_rows, n_cols)
        :param w_init: Initial weight magnitude (if specifiers call for weight initialization)
        :param force_shared_parameters: Use shared parameters for conv layer (allows training).
        :param rng: Random Number Generator (or None)
        :return: A ConvNet
        """
        rng = get_rng(rng)
        n_maps, n_rows, n_cols = input_shape if input_shape is not None else (None, None, None)
        layers = OrderedDict()
        if isinstance(specifiers, (list, tuple)):
            specifiers = OrderedDict((str(i), val) for i, val in enumerate(specifiers))
        for spec_name, spec in specifiers.iteritems():
            if isinstance(spec, ConvInitSpec):
                assert n_maps is not None
                spec = ConvolverSpec(
                    w=w_init*rng.randn(spec.n_maps, n_maps, spec.filter_size[0], spec.filter_size[1]),
                    b=np.zeros(spec.n_maps) if spec.use_bias else False,
                    mode = spec.mode
                    )
            if isinstance(spec, ConvolverSpec):
                n_maps = spec.w.shape[0]
            layers[spec_name] = specifier_to_layer(spec, force_shared_parameters=force_shared_parameters, rng=rng)
        return ConvNet(layers)

    @property
    def parameters(self):
        return sum([l.parameters if isinstance(l, IParameterized) else [] for l in self.layers.values()], [])

    def to_spec(self):
        return ConvNetSpec(OrderedDict((layer_name, lay.to_spec()) for layer_name, lay in self.layers.iteritems()))

    @classmethod
    def from_spec(cls, spec):
        if isinstance(spec, OrderedDict): # "old" format
            return ConvNet.from_init(spec)
        else:
            return ConvNet.from_init(spec.layer_ordered_dict)


def specifier_to_layer(spec, force_shared_parameters=True, rng = None):
    # TODO: Remove, replace with from_spec
    return {
        ConvolverSpec: lambda: ConvLayer(
            w=spec.w,
            b=spec.b,
            force_shared_parameters=force_shared_parameters,
            border_mode= {'full': 0, 'same': 'half', 'valid': 0}[spec.mode] if spec.mode in ('full', 'same', 'valid') else spec.mode,
            filter_flip=False
            ),
        NonlinearitySpec: lambda: Nonlinearity(spec.func),
        PoolerSpec: lambda: Pooler(region=spec.region, stride=spec.stride, mode=spec.mode),
        DropoutSpec: lambda: DropoutLayer(spec.dropout_rate, rng=rng),
        FullyConnectedSpec: lambda: FullyConnectedLayer(w=spec.w, b=spec.b)
        }[spec.__class__]()


def normalize_convnet(convnet, inputs):
    """
    Change the convnet in-place, such that the outputs of convolutions have a standard deviation of 1.

    :param convnet:
    :param inputs:
    :return:
    """
    activations = convnet.get_named_layer_activations.compile()(inputs)

    cum_scale = 1
    for name, act in activations.iteritems():
        if isinstance(convnet.layers[name], ConvLayer):
            this_std = np.std(act)
            cum_scale = this_std / cum_scale
            convnet.layers[name].w.set_value(convnet.layers[name].w.get_value()/cum_scale)
            convnet.layers[name].b.set_value(convnet.layers[name].b.get_value()/this_std)


def spec_to_object(spec):  # Temporary measure... will do this more clanly later.
    cls = {
        ConvNetSpec: ConvNet,
        NonlinearitySpec: Nonlinearity,
        PoolerSpec: Pooler,
        FullyConnectedSpec: FullyConnectedLayer,
        ConvolverSpec: ConvLayer,
        DropoutLayer: DropoutLayer
        }[spec.__class__]\

    return cls.from_spec(spec)
