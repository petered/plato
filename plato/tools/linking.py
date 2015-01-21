from plato.interfaces.decorators import symbolic_standard
from plato.interfaces.interfaces import IParameterized

__author__ = 'peter'


@symbolic_standard
class Chain(IParameterized):

    def __init__(self, *processors):
        self._processors = processors

    def __call__(self, *args):
        out = args
        updates = []
        for p in self._processors:
            out, these_updates = p.symbolic_standard(*out)
            updates += these_updates
        return out, updates

    @property
    def parameters(self):
        return sum([p.parameters for p in self._processors if isinstance(p, IParameterized)], [])
