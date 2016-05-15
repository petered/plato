from collections import OrderedDict, namedtuple
from general.numpy_helpers import get_rng
from plato.core import symbolic, create_shared_variable
from plato.interfaces.helpers import get_named_activation_function
from plato.interfaces.interfaces import IParameterized
import theano.tensor as tt
from theano.tensor.signal.pool import pool_2d
import numpy as np
__author__ = 'peter'


@symbolic
class ConvLayer(IParameterized):

    def __init__(self, w, b, force_shared_parameters = True, border_mode = 'valid', filter_flip = True):
        """
        w is the kernel, an ndarray of shape (n_output_maps, n_input_maps, w_size_y, w_size_x)
        b is the bias, an ndarray of shape (n_output_maps, )
        force_shared_parameters: Set to true if you want to make the parameters shared variables.  If False, the
            parameters will be
        :param border_mode: {'valid', 'full', 'half', int, (int1, int2)}.  Afects
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


@symbolic
class Nonlinearity(object):

    def __init__(self, activation):
        """
        activation:  a name for the activation function. {'relu', 'sig', 'tanh', ...}
        """
        self.activation = get_named_activation_function(activation) if isinstance(activation, basestring) else activation

    def __call__(self, x):
        return self.activation(x)


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
            # print 'Layer %s: %s' % (name, x.ishape)
        return named_activations

    @staticmethod
    def from_init(specifiers, input_shape, w_init=0.01, rng=None):
        rng = get_rng(rng)
        n_maps, n_rows, n_cols = input_shape
        layers = []
        for spec in specifiers:
            instantiated_spec = ConvolverSpec(w=w_init*rng.randn(spec.n_maps, n_maps, spec.filter_size[0], spec.filter_size[1]), b=np.zeros(spec.n_maps), mode = spec.mode) \
                if isinstance(spec, ConvInitSpecifier) else spec
            if isinstance(instantiated_spec, ConvolverSpec):
                n_maps = instantiated_spec.w.shape[0]
                if spec.mode == 'valid':
                    n_rows += -instantiated_spec.w.shape[2] + 1
                    n_cols += -instantiated_spec.w.shape[3] + 1
            elif isinstance(instantiated_spec, PoolerSpec):
                n_rows /= instantiated_spec.region
                n_cols /= instantiated_spec.region
            layers.append(specifier_to_layer(instantiated_spec))
            print '%s output: %s' % (spec, (n_maps, n_rows, n_cols))
        return ConvNet(layers)

    @property
    def parameters(self):
        return sum([l.parameters if isinstance(l, IParameterized) else [] for l in self.layers.values()], [])

def softmax(x, axis=1):
    e_x = tt.exp(x - x.max(axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)


def specifier_to_layer(spec, force_shared_parameters=True):
    return {
        ConvolverSpec: lambda: ConvLayer(
            w=spec.w,
            b=spec.b,
            force_shared_parameters=force_shared_parameters,
            border_mode= {'full': 0, 'same': 1, 'valid': 0}[spec.mode],
            filter_flip=False
            ),
        NonlinearitySpec: lambda: softmax if spec.type=='softmax' else Nonlinearity(spec.type),
        PoolerSpec: lambda: Pooler(region=spec.region, stride=spec.stride, mode=spec.mode)
        }[spec.__class__]()


def layer_to_specifier(spec, )


ConvInitSpecifier = namedtuple('ConvInitSpecifier', ['n_maps', 'filter_size', 'mode'])
NonlinearitySpec = namedtuple('NonlinearitySpec', ['type'])
ConvolverSpec = namedtuple('ConvolverSpec', ['w', 'b', 'mode'])
PoolerSpec = namedtuple('MaxPoolerSpec', ['region', 'stride', 'mode'])

