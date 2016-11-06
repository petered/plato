__author__ = 'peter'


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



def kwarg_map(element_constructor, **kwarg_lists):
    """
    A helper function for when you want to construct a chain of objects with individual arguments for each one.  Can
    be easier to read than a list expansion.

    :param element_constructor: A function of the form object = fcn(**kwargs)
    :param kwarg_lists: A dict of lists, where the index identifies two which element its corresponding value will go.
    :return: A list of objects.

    e.g. Initializing a chain of layers:
        layer_sizes = [784, 240, 240, 10]
        layers = kwarg_map(
            Layer,
            n_in = layer_sizes[:-1],
            n_out = layer_sizes[1:],
            activation = ['tanh', 'tanh', 'softmax'],
            )

    is equivalent to:
        layers = [Layer(n_in=784, n_out=240, activation='tanh'), Layer(n_in=240, n_out=240, activation='tanh'), Layer(n_in=240, n_out=10, activation='softmax')]
    or
        layers = [Layer(n_in=n_in, n_out=n_out, activation=activation) for n_in, n_out, activation in zip(layer_sizes[:-1], layer_sizes[1:], ['tanh', 'tanh', 'softmax'])]

    """
    all_lens = [len(v) for v in kwarg_lists.values()]
    assert len(kwarg_lists)>0, "You need to specify at least list of arguments (otherwise you don't need this function)"
    n_elements = all_lens[0]
    assert all(n_elements == le for le in all_lens), 'Inconsistent lengths: %s' % (all_lens, )
    return [element_constructor(**{k: v[i] for k, v in kwarg_lists.iteritems()}) for i in xrange(n_elements)]
