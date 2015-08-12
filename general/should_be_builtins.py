__author__ = 'peter'

all_equal = lambda *args: all(a == args[0] for a in args[1:])


def bad_value(value, explanation = None):
    """
    :param value: Raise ValueError.  Useful when doing conditional assignment.
    e.g.
    dutch_hand = 'links' if eng_hand=='left' else 'rechts' if eng_hand=='right' else bad_value(eng_hand)
    """
    raise ValueError('Bad Value: %s%s' % (value, ': '+explanation if explanation is not None else ''))


def memoize(fcn):
    """
    Use this to decorate a function whose results you want to cache.
    """
    lookup = {}

    def memoization_wrapper(*args, **kwargs):
        arg_signature = args + ('5243643254_kwargs_start_here', ) + tuple((k, kwargs[k]) for k in sorted(kwargs.keys()))
        if arg_signature in lookup:
            return lookup[arg_signature]
        else:
            out = fcn(*args, **kwargs)
            lookup[arg_signature]=out
            return out

    return memoization_wrapper
