import theano.tensor as tt

__author__ = 'peter'


def softmax(x, axis):
    """ Theano didn't implement softmax very nicely so we have to do some reshaping. """
    e_x = tt.exp(x-x.max(axis=axis, keepdims = True))
    out = e_x/e_x.sum(axis=axis, keepdims=True)
    return out
