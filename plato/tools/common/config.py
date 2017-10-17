from contextlib import contextmanager
import theano

__author__ = 'peter'


@contextmanager
def hold_float_precision(value):
    """
    Change the theano float precesion variable (theano.config.floatX) for all code in a context.  Temporarily overrides
    the value defined in .theanorc.

    Usage:

        with float_preceision('float64'):
            a = create_shared_variable(np.zeros((3, 4)))
            b = a + .....

    :param value: Currently either 'float32' or 'float64'
    """
    if isinstance(value, int):
        value = {32: 'float32', 64: 'float64'}[value]
    assert value in ('float32', 'float64'), "Precision must be 'float32' or 'float64', not '%s'" % (value, )
    old_precision = theano.config.floatX
    theano.config.floatX = value
    yield
    theano.config.floatX = old_precision


float_precision = hold_float_precision # Back-compatibility


@contextmanager
def hold_theano_optimizer(value):
    if value is None:
        value = 'None'
    old_val = theano.config.optimizer
    theano.config.optimizer = value
    yield
    theano.config.optimizer = old_val
