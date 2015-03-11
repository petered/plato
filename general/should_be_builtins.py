__author__ = 'peter'

all_equal = lambda *args: all(a == args[0] for a in args[1:])


def bad_value(value):
    raise ValueError('Bad Value: %s' % (value, ))