from artemis.fileman.primitive_specifiers import PrimativeSpecifier
from artemis.general.should_be_builtins import bad_value

__author__ = 'peter'


class ConvInitSpec(PrimativeSpecifier):

    def __init__(self, n_maps, filter_size, mode, use_bias = True):
        assert len(filter_size)==2, 'Filter size must be (n_rows, n_cols)'
        assert isinstance(n_maps, int)
        assert mode in ('same', 'valid', 'full')
        self.n_maps = n_maps
        self.filter_size = filter_size
        self.mode = mode
        self.use_bias = use_bias

    def shape_transfer(self, (n_samples, n_maps, size_y, size_x)):
        return (n_samples, self.n_maps)+{
            'same': (size_y, size_x),
            'valid': (size_y-self.filter_size[0]+1, size_x-self.filter_size[1]+1),
            'full': (size_y+self.filter_size[0]-1, size_x+self.filter_size[1]-1)
            }[self.mode]


class NonlinearitySpec(PrimativeSpecifier):

    def __init__(self, func):
        assert isinstance(func, basestring), 'func must be a string identifying the nonlinearity.. eg "relu".  Got %s' % (func, )
        self.func = func

    def shape_transfer(self, shape):
        return shape


class ConvolverSpec(PrimativeSpecifier):

    def __init__(self, w, b, mode):
        """
        :param w: A shape (n_output_maps, n_input_maps, n_rows, n_cols) convolutional kernel.  The kernels can be directly
            used for cross-correlation... that is, it is assumed that they are NOT flipped before sliding over the image.
        :param b: A bias of shape (n_output_maps, )
        :param mode: The mode: 'same', 'valid', or 'full', or an integer, in which case it is interpreted as the padding
            for a 'valid' convolution.
        :return:
        """
        assert w.ndim==4
        assert b is False or (b.ndim==1 and w.shape[0] == len(b)), "Number of output maps must match"
        assert isinstance(mode, int) or mode in ('same', 'valid', 'full'), 'Mode "%s" not allowed' % (mode, )
        self.w=w
        self.b=b
        self.mode = mode

    def shape_transfer(self, (n_samples, n_maps, size_y, size_x)):
        return (n_samples, self.w.shape[0])+{
            'same': (size_y, size_x),
            'valid': (size_y-self.w.shape[2]+1, size_x-self.w.shape[3]+1),
            'full': (size_y+self.w.shape[2]-1, size_x+self.w.shape[3]-1)
            }[self.mode]


class PoolerSpec(PrimativeSpecifier):

    def __init__(self, region, mode, stride=None):
        if isinstance(region, int):
            region = (region, region)
        assert mode in ('max', 'average')
        if stride is None:
            stride=region
        elif isinstance(stride, int):
            stride = (stride, stride)
        elif isinstance(stride, tuple):
            assert len(stride)==2
        else:
            bad_value(stride, "Expected None, and int, or a tuple of length 2.  Not %s" % (stride, ))
        self.region = region
        self.stride = stride
        self.mode = mode

    def shape_transfer(self, (n_samples, n_maps, size_y, size_x)):
        return n_samples, n_maps, size_y/self.stride[0], size_x/self.stride[1]


class DropoutSpec(PrimativeSpecifier):

    def __init__(self, dropout_rate):
        assert 0 <= dropout_rate < 1
        self.dropout_rate = dropout_rate

    def shape_transfer(self, (n_samples, n_maps, size_y, size_x)):
        return n_samples, n_maps, size_y, size_x


class FullyConnectedSpec(PrimativeSpecifier):

    def __init__(self, w, b):
        """
        :param w: A shape (n_inputs, n_outputs) Weight matrix
        :param b: A bias of shape (n_outputs, ), or False if no bias is used.
        """
        assert w.ndim==2
        assert b is False or (b.ndim==1 and w.shape[1] == len(b)), "Number of output maps must match"
        self.w=w
        self.b=b

    def shape_transfer(self, input_shape):
        if len(input_shape)==4:
            n_samples, n_maps, size_y, size_x = input_shape
            assert n_maps*size_y*size_x == self.w.shape[0]
            return n_samples, self.w.shape[1], 1, 1
        elif len(input_shape)==2:
            n_samples, input_dims = input_shape
            return n_samples, self.w.shape[1]


class ConvNetSpec(PrimativeSpecifier):

    def __init__(self, layer_ordered_dict):
        self.layer_ordered_dict = layer_ordered_dict

    def shape_transfer(self):
        raise NotImplementedError()




# class ConvNetSpec