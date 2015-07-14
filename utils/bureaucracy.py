__author__ = 'peter'
import numpy as np


def multichannel(fcn):
    """
    Take a function that accepts N args and returns a single argument, and wrap it as a
    function that takes a tuple of length N as an argument and returns a tuple of length
    1.

    This is a temporary thing until we have a proper decorator system for these array
    functions.

    :param fcn: A function of the form
        out_arr = fcn(in_arr_0, in_arr_1, ...)
    :return: A function of the form
        (out_arr, ) = fcn((in_arr_0, in_arr_1, ...))
    """
    return lambda args: (fcn(*args), )


def minibatch_iterate(data, minibatch_size, n_epochs=1):
    """
    Yields minibatches in sequence.
    :param data: A (n_samples, ...) data array
    :param minibatch_size: The number of samples per minibatch
    :param n_epochs: The number of epochs to run for
    :yield: (minibatch_size, ...) data arrays.
    """
    end = len(data)*n_epochs
    ixs = np.arange(minibatch_size)
    while ixs[0] < end:
        yield data[ixs % len(data)]
        ixs+=minibatch_size


def zip_minibatch_iterate(arrays, minibatch_size, n_epochs=1):
    """
    Yields minibatches from each array in arrays in sequence.
    :param arrays: A collection of arrays, all of which must have the same shape[0]
    :param minibatch_size: The number of samples per minibatch
    :param n_epochs: The number of epochs to run for
    :yield: len(arrays) arrays, each of shape: (minibatch_size, )+arr.shape[1:]
    """
    assert isinstance(arrays, (list, tuple)), 'Need at least one array' and len(arrays)>0
    total_size = arrays[0].shape[0]
    assert all(a.shape[0] == total_size for a in arrays), 'All arrays must have the same length!  Lengths are: %s' % ([len(arr) for arr in arrays])
    end = total_size*n_epochs
    ixs = np.arange(minibatch_size)
    while ixs[0] < end:
        yield tuple(a[ixs % total_size] for a in arrays)
        ixs+=minibatch_size


def kwarg_map(element_constructor, **kwarg_lists):
    """
    A helper function for when you want to construct a chain of objects with individual arguments for each one.  Can
    be easier to read than a list expansion.

    :param element_constructor: A function of the form object = fcn(**kwargs)
    :param kwarg_lists: A dict of lists, where the index identifies two which element its corresponding value will go.
    :return: A list of objects.

    e.g. Initializing a chain of layers.

    layer_sizes = [784, 240, 240, 10]
    layers = kwarg_map(
        lambda n_in, n_out, activation: Layer(n_in, n_out, activation),
        n_in = layer_sizes[:-1],
        n_out = layer_sizes[1:],
        activation = ['tanh', 'tanh', 'softmax'],
        )
    """
    all_lens = [len(v) for v in kwarg_lists.values()]
    assert len(kwarg_lists)>0, "You need to specify at least list of arguments (otherwise you don't need this function)"
    n_elements = all_lens[0]
    assert all(n_elements == le for le in all_lens), 'Inconsistent lengths: %s' % (all_lens, )
    return [element_constructor(**{k: v[i] for k, v in kwarg_lists.iteritems()}) for i in xrange(n_elements)]


def create_object_chain_complex(element_constructor, shared_args = {}, each_args = {}, paired_args = {}, n_elements = None):
    """
    * Slated for deletion unless we have use case.  See kwarg_map and see if that doesn't cover your needs.

    A helper function for when you want to construct a chain of objects.  Objects in the chain can either get
    the same arguments or individual arguments.

    :param element_constructor: A function of the form object = fcn(**kwargs)
    :param shared_args: A dict of kwargs that will be passed to all objects
    :param each_args: A dict of lists, where the index identifies two which element its corresponding value will go.
    :param paired_args: A dist of form {(arg_name_1, arg_name_2): [v1, v2, ...]}.  The length of the lists in this dictionary
        is one greater than the number of elements.  See example below for how this is used.
    :return: A list of objects.

    e.g. Initializing a chain of layers.

    layers = kwarg_map(
        element_constructor = lambda n_in, n_out, activation: Layer(n_in, n_out, activation, rng),
        shared_args = dict(rng = np.random.RandomState(1234)),
        each_args = dict(activation = ['tanh', 'tanh', 'softmax'],
        paired_args = dict(('n_in', 'n_out'): [784, 240, 240, 10])
        )
    """

    all_lens = [len(v) for v in each_args.values()]+[len(v)-1 for v in paired_args.values()]
    assert len(all_lens)>0, 'You need to specify at least one "each_args" or "paired args'
    if n_elements is None:
        n_elements = all_lens[0]
    assert all(n_elements == le for le in all_lens), 'Inconsistent lengths: %s' % (all_lens)
    return [element_constructor(
            **dict(
                shared_args.items() +
                {k: v[i] for k, v in each_args.iteritems()}.items() +
                {k0: v[i] for (k0, k1), v in paired_args.iteritems()}.items() +
                {k1: v[i+1] for (k0, k1), v in paired_args.iteritems()}.items())
            ) for i in xrange(n_elements)]
