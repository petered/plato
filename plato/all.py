"""
Import this file to use anything from plato without having to think about where it's located.
"""
from plato.core import symbolic, symbolic_multi, symbolic_simple, symbolic_stateless, symbolic_updater, SymbolicFunction, \
    tdb_trace
from plato.tools.all import *
from plato.tools.misc.tdb_plotting import tdbplot


if __name__ == '__main__':
    for k in sorted(locals().keys()):
        if not k.startswith('__'):
            print '%s: %s' % (k, locals()[k])
