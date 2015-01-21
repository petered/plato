import inspect
from abc import abstractproperty, abstractmethod
from theano.compile.sharedvalue import SharedVariable
from theano.gof.graph import Variable
import theano.tensor as ts
from theano.tensor.type import TensorType
import theano
import numpy as np

"""
It is better not to look at the things happening in here.  It's beautiful on the outside but not on the inside.
All you need to know is this:

You can decorate a symbolic function, method, or callable class with the following decorators:

@symbolic_stateless: If the function does not return state updates.
@symbolic_updater: If the function returns only updates.
@symbolic_standard: If the function returns (outputs, updates) as a tuple.

A decorated function has the following attributes:
symbolic_standard: Return a version of the function in the standard format of out, updates = fcn(*inputs)
compiled: Return a compiled version of the function that will accept and return numpy arrays.
"""

__author__ = 'peter'


class ISymbolicFunction(object):

    def compile(self):
        """
        :return: A version of the fcntion that takes and returns numpy arrays.
        """

    @abstractproperty
    def symbolic_stateless(self):
        """
        :return: a function of the form:
        out = fcn(in_0, in_1, ...)
        Where out and in are tensors.  If the function cannot be cast to this form (for instance because it returns
        multiple outputs or updates) an exception will be raised when it is called.
        """

    @abstractproperty
    def symbolic_standard(self):
        """
        A fcntion of the form:
        (out_0, out_1, ...), ((shared_0, new_shared_0), (shared_1, new_shared_1), ...) = fcn(in_0, in_1, ...)
        """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """ Call the fcntion directly """


class BaseSymbolicFunction(ISymbolicFunction):

    def __init__(self, fcn, instance = None):

        self._fcn = fcn
        self._instance = instance

    def _assert_is_tensor(self, arg, name):
        if not isinstance(arg, Variable):
            raise SymbolicFormatError('%s of function %s should have been a tensor, but was %s' % (name, self._fcn, arg))

    def _assert_all_tensors(self, args, name):
        if not (isinstance(args, tuple) and all(isinstance(arg, Variable) for arg in args)):
            raise SymbolicFormatError('%s of %s must a tuple of tensors.  They were %s instead' % (name, self._fcn, args, ))

    def _assert_all_updates(self, updates):
        if not (isinstance(updates, list) and all(len(up)==2 for up in updates) and
                all(isinstance(old, SharedVariable) and isinstance(new, Variable) for old, new in updates)):
            raise SymbolicFormatError('Updates from %s must be a list of 2-tuples of (shared_variable, update_tensor).  It was %s instead' % (self._fcn, updates, ))

    def _assert_standard_return(self, return_val):
        if not (isinstance(return_val, tuple) and len(return_val)==2):
            raise SymbolicFormatError('Function %s was expected to return a 2-tuple of (outputs, updates) but returned %s instead' % (self._fcn, return_val))
        outputs, updates = return_val
        self._assert_all_tensors(outputs, 'Outputs')
        self._assert_all_updates(updates)

    def __get__(self, instance, owner):
        return self.__class__(self._fcn, instance=instance)

    def compile(self):
        return AutoCompilingFunction(self)

    def _call_fcn(self, *args, **kwargs):
        return self._fcn(*args, **kwargs) if self._instance is None else self._fcn(self._instance, *args, **kwargs)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class SymbolicStatelessFunction(BaseSymbolicFunction):
    """
    Use this to decorate a symbolic fcntion of the form:
    out = fcn(in_0, in_1, ...)    OR
    (out_0, out_1, ...) = fcn(in_0, in_1, ...)
    """

    def __call__(self, *args):
        self._assert_all_tensors(args, 'Arguments')
        out = self._call_fcn(*args)
        self._assert_is_tensor(out, 'Output')
        return out

    @property
    def symbolic_stateless(self):
        return self

    @property
    def symbolic_standard(self):
        return SymbolicStandardFunction(self._standard_function)

    def _standard_function(self, *args, **kwargs):
        out = self._call_fcn(*args, **kwargs)
        return (out, ), []


class SymbolicStandardFunction(BaseSymbolicFunction):

    def __call__(self, *args):
        self._assert_all_tensors(args, 'Arguments')
        return_val = self._call_fcn(*args)
        self._assert_standard_return(return_val)
        return return_val

    @property
    def symbolic_standard(self):
        return self

    @property
    def symbolic_stateless(self):
        return SymbolicStatelessFunction(self._stateless_function)

    def _stateless_function(self, *args, **kwargs):
        outputs, updates = self._fcn(*args, **kwargs)
        assert len(updates)==0, "You tried to call %s as a stateless function, but it returns updates, so this can't be done." \
            % self._fcn
        assert len(outputs)==1, "You tried to call %s as a stateless function, but it returns multiple outputs, so this can't be done." \
            % self._fcn
        out, = outputs
        return out


class SymbolicUpdateFunction(BaseSymbolicFunction):

    def __call__(self, *args, **kwargs):
        self._assert_all_tensors(args, 'Arguments')
        updates = self._call_fcn(*args, **kwargs)
        self._assert_all_updates(updates)
        return updates

    @property
    def symbolic_stateless(self):
        raise Exception("Tried to get the symbolic_stateless function from an %s\n, which is a SymbolicUpdateFunction. - "
            "This won't work because updaters have state.")

    @property
    def symbolic_standard(self):
        return SymbolicStandardFunction(lambda *args: _standardize_update_fcn(self._fcn, *args), instance = self._instance)


# symbolic_stateless = SymbolicStatelessFunction
# symbolic_standard = SymbolicStandardFunction
# symbolic_updater = SymbolicUpdateFunction

# Cases to consider:
# 1) Function: called directly with instance = None
# 2) Method: Called from __get__ when the method is requested.  instance is the object to which the method is bound
# 3) Callable class:

def symbolic_stateless(fcn):
    return _decorate_anything(SymbolicStatelessFunction, fcn)


def symbolic_standard(fcn):
    return _decorate_anything(SymbolicStandardFunction, fcn)


def symbolic_updater(fcn):
    return _decorate_anything(SymbolicUpdateFunction, fcn)


def _decorate_anything(function_type, thing):
    if inspect.isclass(thing): # Case 3: Class
        return _decorate_callable_class(function_type = function_type, callable_class = thing)
    else:  # Cases 1 and 2: Function or method
        return function_type(thing)


def _decorate_callable_class(function_type, callable_class):

    assert hasattr(callable_class, '__call__')

    callable_class.__call__ = function_type(callable_class.__call__)

    def new_constructor(*args, **kwargs):
        obj = callable_class(*args, **kwargs)
        cls = obj.__class__
        obj.__class__ = cls.__class__(cls.__name__ + "With"+function_type.__class__.__name__, (cls, function_type), {})
        function_type.__init__(obj, fcn = obj, instance = None)
        return obj

    # callable_class.__new__ = new_constructor

    return new_constructor


class SymbolicFormatError(Exception):
    pass


# def _standardize_stateless_fcn(fcn, args):
#     out = fcn(*args)
#     if not isinstance(out, tuple):
#         out = (out, )
#     return out, []
#
#
# def _standardize_update_fcn(fcn, args):
#     updates = fcn(*args)
#     return (), updates


class AutoCompilingFunction(object):
    """
    Given a Symbolic function, turn it into a compiled fcniton that will accept and return numpy arrays.

    Actual compilation happens on the first use of the fcntion, since it needs to see the arguments in order to
    instantiate the input tensors.
    """

    def __init__(self, fcn, cast_floats_to_floatX = True):

        assert isinstance(fcn, ISymbolicFunction), 'You must pass a symbolic fcniton.  Decorate it!'
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
            tensor_args = [_data_to_tensor(arg, cast_floats_to_floatx = self._cast_floats_to_floatX) for arg in args]
            return_value = self._fcn(*tensor_args)
            if isinstance(self._fcn, SymbolicStatelessFunction):
                outputs = return_value
                updates = []
            elif isinstance(self._fcn, SymbolicStandardFunction):
                outputs, updates = return_value
            elif isinstance(self._fcn, SymbolicUpdateFunction):
                outputs = ()
                updates = return_value
            else:
                raise Exception("Get OUT!")
            self._compiled_fcn = theano.function(inputs = tensor_args, outputs = outputs, updates = updates)
        return self._compiled_fcn(*args)


def _is_symbol_or_value(var):
    return isinstance(var, ts.TensorType) or isinstance(var, np.ndarray) or np.isscalar(var)


def _data_to_tensor(data, name = None, cast_floats_to_floatx = True):
    ndim = 0 if np.isscalar(data) else data.ndim
    dtype = theano.config.floatX if (cast_floats_to_floatx and (isinstance(data, float) or isinstance(data, np.ndarray) and data.dtype == 'float')) \
        else 'int64' if isinstance(data, int) \
        else 'float64' if isinstance(data, float) \
        else data.dtype
    return TensorType(dtype, (None, )*ndim)(name)
