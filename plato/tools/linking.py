from __builtin__ import property
from plato.interfaces.decorators import symbolic_standard
from plato.interfaces.interfaces import IParameterized

__author__ = 'peter'


@symbolic_standard
class Chain(IParameterized):
    """
    A composition of symbolic functions:

    Chain(f, g, h)(x) is f(g(h(x)))

    Details:
    Chain calls functions in the standard format:
        out, updates = func(inputs)
        (Any symbolic-decorated function can becalled like this)

    Chain can be called in the standard format:
        outputs, updates = my_chain(inputs)  # or
        outputs, updates = my_chain.symbolic_standard(inputs)

    If none of the internals return updates, and the last function in the chain
    returns just a single output, Chain can also be called in the symbolic stateless
    format:

        output = my_chain.symbolic_stateless(input_0, input_1, ...)

    If, however, internals return updates, or the last function returns multiple
    updates, this will raise an Exception.
    """

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


@symbolic_standard
class Branch(IParameterized):

    def __init__(self, *processors):
        self._processors = processors

    def __call__(self, x):
        outputs = tuple(p.symbolic_stateless(x) for p in self._processors)
        return outputs, []

    @property
    def parameters(self):
        return sum([p.parameters for p in self._processors if isinstance(p, IParameterized)], [])
