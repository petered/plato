from plato.core import symbolic_multi
from plato.interfaces.interfaces import IParameterized

__author__ = 'peter'


@symbolic_multi
class Chain(IParameterized):
    """
    A composition of symbolic functions:

    Chain(f, g, h)(x) is f(g(h(x)))

    tuple_of_output_tensors = my_chain(input_tensor_0, input_tensor_1, ...)

    If the last function in the chain returns just a single output, Chain can also be called in the
    symbolic_simple format:
        output_tensor = my_chain.symbolic_simple(input_0, input_1, ...)
    If, however, the last function returns multiple updates, this will raise an Exception.
    """

    def __init__(self, *processors):
        self._processors = processors

    def __call__(self, *args):
        out = args
        for p in self._processors:
            out = p.to_format(symbolic_multi)(*out)
        return out

    @property
    def parameters(self):
        return sum([p.parameters for p in self._processors if isinstance(p, IParameterized)], [])


@symbolic_multi
class Branch(IParameterized):
    """
    Given a set of N One-in-one-out processors, make a composite processor
    that takes one in and N out.
    """

    def __init__(self, *processors):
        self._processors = processors

    def __call__(self, x):
        results = tuple(p.to_format(symbolic_multi)(x) for p in self._processors)
        outputs = sum([o for o in results], ())
        return outputs

    @property
    def parameters(self):
        return sum([p.parameters for p in self._processors if isinstance(p, IParameterized)], [])
