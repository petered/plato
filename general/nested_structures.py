import numpy as np

__author__ = 'peter'


def flatten_struct(struct, primatives = (int, float, np.ndarray, basestring, bool)):
    """
    Given some nested struct, return an OrderedDict<str: primative>, where primative
    is some some kind of object that you don't break down any further, and str is a
    string representation of how you would access that propery from the root object.

    Don't try any fancy circular references here, it's not going to go well for you.

    :param struct: Something, anything.
    :param primatives: A list of classes that will not be broken into.
    :return: OrderedDict<str: primative>
    """

    if isinstance(struct, primatives):
        return [(None, struct)]
    elif isinstance(struct, dict):
        return sum([
            [("[%s]%s" % (("'%s'" % key if isinstance(key, str) else key), subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value)]
                for key, value in struct.iteritems()
            ], [])
    elif isinstance(struct, (list, tuple)):
        # for i, value in enumerate(struct):
        return sum([
            [("[%s]%s" % (i, subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value)]
                for i, value in enumerate(struct)
            ], [])
    else:  # It's some kind of object, lets break it down.
        return sum([
            [(".%s%s" % (key, subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value)]
                for key, value in struct.__dict__.iteritems()
            ], [])
