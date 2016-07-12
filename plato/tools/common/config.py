from contextlib import contextmanager
import theano

__author__ = 'peter'

@contextmanager
def float_precision(value):
    assert value in ('float32', 'float64'), "Precision must be 'float32' or 'float64', not '%s'" % (value, )
    old_precision = theano.config.floatX
    theano.config.floatX = value
    yield
    theano.config.floatX = old_precision


# with float_precision('float64'):
#     # Create your function...
