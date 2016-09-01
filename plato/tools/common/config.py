from contextlib import contextmanager
import theano

__author__ = 'peter'

@contextmanager
def float_precision(value):
    """
    Change the theano float precesion variable (theano.config.floatX) for all code in a context.  Temporarily overrides
    the value defined in .theanorc.

    Usage:

        with float_preceision('float64'):
            a = create_shared_variable(np.zeros((3, 4)))
            b = a + .....

    :param value: Currently either 'float32' or 'float64'
    """
    assert value in ('float32', 'float64'), "Precision must be 'float32' or 'float64', not '%s'" % (value, )
    old_precision = theano.config.floatX
    theano.config.floatX = value
    yield
    theano.config.floatX = old_precision
