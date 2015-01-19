from plato.interfaces.interfaces import IParameterized

__author__ = 'peter'


class Chain(IParameterized):

    def __init__(self, *processors):
        self._processors = processors

    def __call__(self, x):
        out = x
        updates = []
        for p in self._processors:
            out, these_updates = p.standard_symbolic_function(out)
            updates += these_updates
        return out, updates

    @property
    def parameters(self):
        return sum([p.parameters for p in self._processors if isinstance(p, IParameterized)], [])
