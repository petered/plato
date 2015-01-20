from abc import abstractproperty, abstractmethod
from theano.compile.sharedvalue import SharedVariable
import theano.tensor as ts
from theano.tensor.type import TensorType
import theano
import numpy as np


__author__ = 'peter'


class ISymbolicFunction(object):

    @abstractproperty
    def compiled(self):
        """
        :return: A version of the function that takes and returns numpy arrays.
        """

    @abstractproperty
    def symbolic_standard(self):
        """
        A function of the form:
        (out_0, out_1, ...), ((shared_0, new_shared_0), (shared_1, new_shared_1), ...) = fcn(in_0, in_1, ...)
        """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Call the function directly """


class SymbolicStatelessFunction(ISymbolicFunction):
    """
    Use this to decorate a symbolic function of the form:
    out = fcn(in_0, in_1, ...)    OR
    (out_0, out_1, ...) = fcn(in_0, in_1, ...)
    """

    def __init__(self, fcn):
        self._fcn = fcn

    def __call__(self, *args):
        assert all(isinstance(arg, TensorType) for arg in args), \
            'Arguments must all be tensors.  They were %s instead' % ([type(a) for a in args], )
        out = self._fcn(*args)
        assert all(isinstance(o, TensorType) for o in out) if type(out) is tuple else isinstance(out, TensorType)
        return out

    @property
    def symbolic_standard(self):
        return SymbolicStandardFunction(lambda *args: _standardize_stateless_fcn(self._fcn, args))

    def compiled(self):
        return AutoCompilingFunction(self)


class SymbolicStandardFunction(ISymbolicFunction):

    def __call__(self, *args):
        assert all(isinstance(arg, TensorType) for arg in args), \
            'Arguments must all be tensors.  They were %s instead' % ([type(a) for a in args], )
        out, updates = self._fcn(*args)
        assert all(isinstance(o, TensorType) for o in out) if type(out) is tuple else isinstance(out, TensorType), \
            'Outputs must a single tensor or a tuple of tensors'
        assert isinstance(updates, list) and all(len(up)==2 for up in updates) and \
            all(isinstance(old, SharedVariable) and isinstance(new, TensorType) for old, new in updates), \
            'Updates must be a list of 2-tuples of (shared_variable, update_tensor).  It was %s instead' % (updates, )
        return out

    def symbolic_standard(self):
        return self

    def compiled(self):
        return AutoCompilingFunction(self)


def symbolic_stateless(fcn):
    return SymbolicStatelessFunction(fcn)


def symbolic_standard(fcn):
    return SymbolicStandardFunction(fcn)


def _standardize_stateless_fcn(fcn, args):
    out = fcn(*args)
    if not isinstance(out, tuple):
        out = (out, )
    return out, []


class AutoCompilingFunction(object):
    """
    Given a Symbolic function, turn it into a compiled funciton that will accept and return numpy arrays.

    Actual compilation happens on the first use of the function, since it needs to see the arguments in order to
    instantiate the input tensors.
    """

    def __init__(self, fcn, cast_floats_to_floatX = True):

        assert isinstance(fcn, ISymbolicFunction), 'You must pass a symbolic funciton.  Decorate it!'
        self._fcn = fcn
        self._format = format
        self._compiled_fcn = None
        self._cast_floats_to_floatX = cast_floats_to_floatX

    def __call__(self, *args):
        """
        :param args, kwargs are the arguments that would go into fcn, but as real numpy arrays instead of symbols
        returns the result, in numpy arrays.
        """
        if self._compiled_fcn is None:
            tensor_args = [_data_to_tensor(arg, cast_floats_to_floatX = self._cast_floats_to_floatX) for arg in args]
            return_value = self._fcn(*tensor_args)
            if isinstance(self._fcn, SymbolicStatelessFunction):
                outputs = return_value
                updates = []
            elif isinstance(self._fcn, SymbolicStandardFunction):
                outputs, updates = return_value
            else:
                raise Exception("Get OUT!")
            self._compiled_fcn = theano.function(inputs = tensor_args, outputs = outputs, updates = updates)


def _is_symbol_or_value(var):
    return isinstance(var, ts.TensorType) or isinstance(var, np.ndarray) or np.isscalar(var)


def _data_to_tensor(data, name = None, cast_floats_to_floatx = True):
    ndim = 0 if np.isscalar(data) else data.ndim
    dtype = theano.config.floatX if (cast_floats_to_floatx and (np.isscalar(data) and type(data) is float) or data.dtype == 'float') else data.dtype
    return TensorType(dtype, (None, )*ndim)(name)
