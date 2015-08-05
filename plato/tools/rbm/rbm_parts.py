__author__ = 'peter'




@symbolic_stateless
class FullyConnectedBridge(IParameterized, IFreeEnergy):
    """
    An element which multiplies the input by some weight matrix w and adds a bias.
    """

    def __init__(self, w, b = 0, b_rev = None, scale = False, normalize_minibatch = False, use_bias = True):
        """
        :param w: Initial weight value.  Can be:
            - A numpy array, in which case a shared variable is instantiated from this data.
            - A symbolic variable that is either a shared variabe or descended from a shared variable.
              This is used when there are shared parameters.
        :param b: Can be:
            - A numpy vector representing the initial bias on the hidden layer, where len(b) = w.shape[1]
            - A scaler, which just initializes the full vector to this value
        :param b_rev: Can be:
            - A numpy vector representing the initial bias on the visible layer, where len(b) = w.shape[0]
            - A scaler, which just initializes the full vector to this value
            - None, in which case b_rev is not created (for instance in an MLP).
        """
        self._w, w_params, w_shape = initialize_param(w, shape = (None, None), name = 'w')
        self._b, b_params, b_shape = initialize_param(b, shape = w_shape[1], name = 'b')
        self._b_rev, b_rev_params, b_rev_shape = initialize_param(b_rev, shape = w_shape[0], name = 'b_rev')
        self._log_scale, log_scale_params, log_scale_shape = initialize_param(0 if scale else None, shape = w.shape[1], name = 'log_scale')
        self._params = w_params+b_params+b_rev_params+log_scale_params
        self._normalize_minibatch = normalize_minibatch
        self._use_bias = use_bias

    def __call__(self, x):
        current = x.flatten(2).dot(self._w)

        if self._normalize_minibatch:
            current = (current - current.mean(axis = 0, keepdims = True)) / (current.std(axis = 0, keepdims = True) + 1e-9)

        if self._log_scale is not None:
            current = current * tt.exp(self._log_scale)

        y = current + self._b if self._use_bias else current
        return y

    @property
    def parameters(self):
        return self._params if self._use_bias else [self._w]

    def reverse(self, y):
        assert self._b_rev is not None, 'You are calling reverse on this bridge, but you failed to specify b_rev.'
        assert not self._normalize_minibatch, "Don't really know about this case..."
        return y.flatten(2).dot(self._w.T)+self._b_rev

    def free_energy(self, visible):
        return -visible.flatten(2).dot(self._b_rev)


@symbolic_stateless
class ConvolutionalBidirectionalBridge(IParameterized, IFreeEnergy):

    def __init__(self, w, b=0, b_rev=None, stride = (1, 1)):
        self._w, w_params, w_shape = initialize_param(w, shape = (None, None, None, None), name = 'w')
        self._b, b_params, b_shape = initialize_param(b, shape = w_shape[0], name = 'b')
        self._b_rev, b_rev_params, b_rev_shape = initialize_param(b_rev, shape = w_shape[1], name = 'b_rev')
        self._params = w_params+b_params+b_rev_params
        self._stride = stride

    def __call__(self, x):
        y = tt.nnet.conv2d(x, self._w, border_mode='valid', subsample = self._stride) + self._b.dimshuffle('x', 0, 'x', 'x')
        return y

    @property
    def parameters(self):
        return self._params

    def reverse(self, y):

        assert self._stride == (1, 1), 'Only support single-step strides for now...'
        # But there's this approach... https://groups.google.com/forum/#!topic/theano-users/Xw4d00iV4yk
        return tt.nnet.conv2d(y, self._w.dimshuffle(1, 0, 2, 3)[:, :, ::-1, ::-1], border_mode='full') + self._b_rev.dimshuffle('x', 0, 'x', 'x')

    def free_energy(self, visible):
        return -tt.sum(visible*self._b_rev.dimshuffle('x', 0, 'x', 'x'), axis = (2, 3))
