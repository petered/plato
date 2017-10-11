from plato.core import add_update, symbolic_simple, create_shared_variable, symbolic
import theano
import theano.tensor as tt
import numpy as np
from plato.interfaces.helpers import shared_like

__author__ = 'peter'


def softmax(x, axis):
    """ Theano didn't implement softmax very nicely so we have to do some reshaping. """
    e_x = tt.exp(x-x.max(axis=axis, keepdims = True))
    out = e_x/e_x.sum(axis=axis, keepdims=True)
    return out


@symbolic
def running_average(data, decay=None, shape=None, initial_value = 0., return_n = False, elementwise=True):
    """
    :param data:
    :param decay: Can be:
        - A number between 0 and 1, in which case 1 indicates "no memory" and lower decay corresponds to longer memery.
        - None, in which case we revert to the "running average" function, which is an average of all points since the start.
    :param shape: Optionally, the shape of the variable, if you know it in advance (passing this allows you to use it in scan loops)
    :param initial_value: The initial value for the running average (only has effect when decay is not None)
    :return:
    """

    if not elementwise:
        data = data.mean()
        shape = ()

    if decay is None:
        assert initial_value == 0, "Initial value has no effect when decay=None.  So don't pass anything for it."
        n_points = theano.shared(np.array(1).astype(theano.config.floatX))
        avg = shared_like(data, dtype='floatX') if shape is None else create_shared_variable(np.zeros(shape))
        new_avg = data*(1./n_points) + avg*(n_points-1.)/n_points
        add_update(avg, new_avg)
        add_update(n_points, n_points+1)
        return (new_avg, n_points) if return_n else new_avg
    else:
        assert return_n is False
        old_avg = shared_like(data, value=initial_value) if shape is None else create_shared_variable(np.zeros(shape)+initial_value, name='running_average{}'.format(shape))
        new_avg = (1-decay)*old_avg + decay*data
        add_update(old_avg, new_avg)
        return new_avg


@symbolic
def running_mean_and_variance(data, decay = None, shape = None, elementwise=True, initial_mean=0., initial_var=0.):
    """
    Compute the running mean and variance of the data.
    Formula from this useful document: http://people.ds.cam.ac.uk/fanf2/hermes/doc/antiforgery/stats.pdf
    (Thank you Tony Finch)
    :param decay: Can be:
        - A number between 0 and 1, in which case 1 indicates "no memory" and lower decay corresponds to longer memery.
        - None, in which case we revert to the "running average" function, which is an average of all points since the start.
    :param shape: Optionally, the shape of the variable, if you know it in advance (passing this allows you to use it in scan loops)
    :return:

    """
    # TODO: Verify that variance estimate is correct when elementwise=False
    if not elementwise:
        shape=()
    if elementwise and (initial_mean!=0 or initial_var!=0):
        assert shape is not None, "Due to construction delays, we are not able to offer shape-free initializations on our elementise line of products yet.  We apologize for the inconvenience."

    s_last = shared_like(data, dtype='floatX', value=initial_var) if shape is None else create_shared_variable(np.zeros(shape)+initial_var)
    mean_last = shared_like(data, dtype='floatX', value=initial_mean) if shape is None else create_shared_variable(np.zeros(shape)+initial_mean)
    if decay is None:
        mean_new, n = running_average(data, shape=shape, return_n=True, elementwise=elementwise, initial_value=initial_mean)
        s_new = s_last + (data-mean_last)*(data-mean_new)
        if not elementwise:
            s_new = s_new.mean()
        var_new = s_new/n
    else:
        mean_new = running_average(data, shape=shape, decay=decay, elementwise=elementwise, initial_value=initial_mean)
        s_new = (1-decay)*(s_last + decay*(data-mean_last)**2)
        if not elementwise:
            s_new = s_new.mean()
        var_new = s_new
    add_update(mean_last, mean_new)
    add_update(s_last, s_new)
    return mean_new, var_new


@symbolic
def running_variance(data, decay=None, shape = None, elementwise=True, initial_value = 0.):
    """
    Compute the running mean and variance of the data.
    :param data: A D
    :param decay:
    :param shape:
    :return:
    """
    return running_mean_and_variance(data=data, decay=decay, shape=shape, elementwise=elementwise, initial_var = initial_value)[1]
