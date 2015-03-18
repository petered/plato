__author__ = 'peter'


def multichannel(fcn):
    """
    Take a function that accepts input arrays.

    This is a temporary thing until we have a proper decorator system for these array
    functions.

    :param fcn: A function of the form
        out_arr = fcn(in_arr_0, in_arr_1, ...)
    :return: A function of the form
        (out_arr, ) = fcn((in_arr_0, in_arr_1, ...))
    """
    return lambda args: (fcn(*args), )
