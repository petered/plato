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

    lookup = {}

    def memoization_wrapper(*args):
        if args in lookup:
            return lookup[args]
        else:
            out = fcn(*args)
            lookup[args]=out
            return out

    return memoization_wrapper
