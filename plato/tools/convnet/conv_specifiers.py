from artemis.fileman.primitive_specifiers import PrimativeSpecifier

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


class NonlinearitySpec(PrimativeSpecifier):

    def __init__(self, func):
        assert isinstance(func, basestring), 'func must be a string identifying the nonlinearity.. eg "relu".  Got %s' % (func, )
        self.func = func


class ConvolverSpec(PrimativeSpecifier):

    def __init__(self, w, b, mode):
        """
        :param w: A shape (n_output_maps, n_input_maps, n_rows, n_cols) convolutional kernel.  It should be assumed that
            weights will be flipped before sliding across the image.
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


class PoolerSpec(PrimativeSpecifier):

    def __init__(self, region, mode, stride=None):
        if isinstance(region, int):
            region = (region, region)
        assert mode in ('max', 'average')
        if stride is None:
            stride=region
        elif isinstance(stride, int):
            stride = (stride, stride)
        self.region = region
        self.stride = stride
        self.mode = mode


class DropoutSpec(PrimativeSpecifier):

    def __init__(self, dropout_rate):
        assert 0 <= dropout_rate < 1
        self.dropout_rate = dropout_rate



#
# def conv_init_spec(n_maps, filter_size, mode):
#     assert len(filter_size)==2, 'Filter size must be (n_rows, n_cols)'
#     assert isinstance(n_maps, int)
#     assert mode in ('same', 'valid', 'full')
#     return dict(type=SpecTypes.CONV_INIT, n_maps=n_maps, filter_size=filter_size, mode=mode)
#
# def nonlinearity_spec(func):
#     return dict(type=SpecTypes.NONLINEARITY, func=func)
#
# def convolver_spec(w, b, mode):
#
#
# def pooler_spec(region, mode, stride=None):
#     if isinstance(region, int):
#         region = (region, region)
#     assert mode in ('max', 'average')
#     if stride is None:
#         stride=region
#     return dict(type=SpecTypes.POOLER, region=region, stride=stride, mode=mode)


# ConvInitSpecifier = namedtuple('ConvInitSpecifier', ['n_maps', 'filter_size', 'mode'])
# NonlinearitySpec = namedtuple('NonlinearitySpec', ['type'])
# ConvolverSpec = namedtuple('ConvolverSpec', ['w', 'b', 'mode'])
# PoolerSpec = namedtuple('MaxPoolerSpec', ['region', 'stride', 'mode'])
