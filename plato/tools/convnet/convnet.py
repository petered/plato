from collections import OrderedDict
from general.numpy_helpers import get_rng
from plato.core import symbolic, create_shared_variable
from plato.interfaces.helpers import get_named_activation_function
from plato.interfaces.interfaces import IParameterized
from plato.tools.convnet.conv_specifiers import ConvInitSpec, ConvolverSpec, PoolerSpec, NonlinearitySpec
import theano.tensor as tt
from theano.tensor.signal.pool import pool_2d
import logging
import numpy as np
__author__ = 'peter'

logging.basicConfig()
LOGGER = logging.getLogger('plato')
LOGGER.setLevel(logging.WARN)

@symbolic
class ConvLayer(IParameterized):

    def __init__(self, w, b, force_shared_parameters = True, border_mode = 'valid', filter_flip = True):
        """
        w is the kernel, an ndarray of shape (n_output_maps, n_input_maps, w_size_y, w_size_x)
        b is the bias, an ndarray of shape (n_output_maps, )
        force_shared_parameters: Set to true if you want to make the parameters shared variables.  If False, the
            parameters will be be constants (which allows for certain optimizations)
        :param border_mode: {'valid', 'full', 'half', int, (int1, int2)}.  Affects
            default is 'valid'.  See theano.tensor.nnet.conv2d docstring for details.
        """
        self.w = create_shared_variable(w) if force_shared_parameters else tt.constant(w)
        self.b = create_shared_variable(b) if force_shared_parameters else tt.constant(b)
        self.border_mode = border_mode
        self.filter_flip = filter_flip

    def __call__(self, x):
        """
        param x: A (n_samples, n_input_maps, size_y, size_x) image/feature tensor
        return: A (n_samples, n_output_maps, size_y-w_size_y+1, size_x-w_size_x+1) tensor
        """
        result = tt.nnet.conv2d(input=x, filters=self.w, border_mode=self.border_mode, filter_flip=self.filter_flip) + self.b[:, None, None]
        return result

    @property
    def parameters(self):
        return [self.w, self.b]
    
    def to_spec(self):
        return ConvolverSpec(self.w.get_value(), self.b.get_value(), self.border_mode)


@symbolic
class Nonlinearity(object):

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


@symbolic
class Pooler(object):

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
    def get_named_layer_activations(self, x):
        """
        :returns: An OrderedDict<layer_name/index, activation>
            If you instantiated the convnet with an OrderedDict, the keys will correspond to the keys for the layers.
            Otherwise, they will correspond to the index which identifies the order of the layer.
        """
        named_activations = OrderedDict()
        for name, layer in self.layers.iteritems():
            x = layer(x)
            named_activations[name] = x
        return named_activations

    @staticmethod
    def from_init(specifiers, input_shape, w_init=0.01, force_shared_parameters = True, rng=None):
        """
        Convenient initialization function.
        :param specifiers:
        :param input_shape:
        :param w_init:
        :param rng:
        :return:
        """
        rng = get_rng(rng)
        n_maps, n_rows, n_cols = input_shape
        layers = OrderedDict()
        if isinstance(specifiers, (list, tuple)):
            specifiers = OrderedDict(enumerate(specifiers))
        for spec_name, spec in specifiers.iteritems():
            if isinstance(spec, ConvInitSpec):
                spec = ConvolverSpec(
                    w=w_init*rng.randn(spec.n_maps, n_maps, spec.filter_size[0], spec.filter_size[1]),
                    b=np.zeros(spec.n_maps),
                    mode = spec.mode
                    )
            if isinstance(spec, ConvolverSpec):
                n_maps = spec.w.shape[0]
                if spec.mode == 'valid':
                    n_rows += -spec.w.shape[2] + 1
                    n_cols += -spec.w.shape[3] + 1
                elif isinstance(spec.mode, int):
                    n_rows += -spec.w.shape[2] + 1 + spec.mode*2
                    n_cols += -spec.w.shape[3] + 1 + spec.mode*2
            elif isinstance(spec, PoolerSpec):
                n_rows /= spec.region[0]
                n_cols /= spec.region[1]
            layers[spec_name] = specifier_to_layer(spec, force_shared_parameters=force_shared_parameters)
            LOGGER.info('Layer "%s" (%s) output shape: %s' % (spec_name, spec.__class__.__name__, (n_maps, n_rows, n_cols)))
        return ConvNet(layers)

    @property
    def parameters(self):
        return sum([l.parameters if isinstance(l, IParameterized) else [] for l in self.layers.values()], [])

    def to_spec(self):
        return OrderedDict((layer_name, lay.to_spec()) for layer_name, lay in self.layers.iteritems())


def specifier_to_layer(spec, force_shared_parameters=True):
    return {
        ConvolverSpec: lambda: ConvLayer(
            w=spec.w,
            b=spec.b,
            force_shared_parameters=force_shared_parameters,
            border_mode= {'full': 0, 'same': 1, 'valid': 0}[spec.mode] if spec.mode in ('full', 'same', 'valid') else spec.mode,
            filter_flip=False
            ),
        NonlinearitySpec: lambda: Nonlinearity(spec.func),
        PoolerSpec: lambda: Pooler(region=spec.region, stride=spec.stride, mode=spec.mode)
        }[spec.__class__]()
