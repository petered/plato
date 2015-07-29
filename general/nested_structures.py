import numpy as np

__author__ = 'peter'


def flatten_struct(struct, primatives = (int, float, np.ndarray, basestring, bool), custom_handlers = {}, memo = None):
    """
    Given some nested struct, return a list<*(str, primative)>, where primative
    is some some kind of object that you don't break down any further, and str is a
    string representation of how you would access that propery from the root object.

    Don't try any fancy circular references here, it's not going to go well for you.

    :param struct: Something, anything.
    :param primatives: A list of classes that will not be broken into.
    :param custum_handlers: A dict<type:func> where func has the form data = func(obj).  These
        will be called if the type of the struct is in the dict of custom handlers.
    :return: list<*(str, primative)>
    """
    if memo is None:
        memo = {}

    if id(struct) in memo:
        return [(None, memo[id(struct)])]
    else:
        memo[id(struct)] = 'Already Seen object at %s' % hex(id(struct))

    if isinstance(struct, primatives):
        return [(None, struct)]
    elif isinstance(struct, tuple(custom_handlers.keys())):
        handler = custom_handlers[custom_handlers.keys()[[isinstance(struct, t) for t in custom_handlers].index(True)]]
        return [(None, handler(struct))]
    elif isinstance(struct, dict):
        return sum([
            [("[%s]%s" % (("'%s'" % key if isinstance(key, str) else key), subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value, custom_handlers=custom_handlers, memo=memo)]
                for key, value in struct.iteritems()
            ], [])
    elif isinstance(struct, (list, tuple)):
        # for i, value in enumerate(struct):
        return sum([
            [("[%s]%s" % (i, subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value, custom_handlers=custom_handlers, memo=memo)]
                for i, value in enumerate(struct)
            ], [])
    elif struct is None or not hasattr(struct, '__dict__'):
        return []
    else:  # It's some kind of object, lets break it down.
        return sum([
            [(".%s%s" % (key, subkey if subkey is not None else ''), v)
                for subkey, v in flatten_struct(value, custom_handlers=custom_handlers, memo=memo)]
                for key, value in struct.__dict__.iteritems()
            ], [])
