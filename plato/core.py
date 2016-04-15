from collections import OrderedDict
from functools import partial
import inspect
import logging
import sys
from general.local_capture import CaptureLocals
from general.nested_structures import flatten_struct, expand_struct
from theano.compile.sharedvalue import SharedVariable
from theano.gof.graph import Variable
import theano.tensor as tt
from theano.tensor.type import TensorType
import theano
import numpy as np
from theano.tensor.var import TensorConstant

"""
This module contains the plato decorators (@symbolic, etc) and their implementations.

You can decorate a symbolic function, method, or callable class with:

@symbolic
def my_symbolic_function(x, y):
    return x+y

You can then compile this function:

f = my_symbolic_function.compile()

And feed it numeric data.

assert f(3, 4) == 7

You can also be more strict, and demand a specific interface from functions:
@symbolic_simple: If the function just returns a single variable and does not update state.
@symbolic_updater: If the function returns only state updates.
@symbolic_standard: If the function returns (outputs, updates) as a tuple.

A decorated function has methods bound to it which allow it to be compiled and called in a standard format.
These methods are described in the ISymbolicFunction interface below.
"""

__author__ = 'peter'

PLATO_LOGGER = logging.getLogger('PlatoLogger')
PLATO_LOGGER.setLevel(logging.WARN)

# Add properties to the "Variable" class (the base class of all symbolic variables), so that you easily inspect
# the initial values that are attached to them.
Variable.ival = property(lambda self: (self.get_value() if isinstance(self, SharedVariable) else self.data if isinstance(self, TensorConstant) else self.tag.test_value))
Variable.ishape = property(lambda self: self.ival.shape)
Variable.indim = property(lambda self: self.ival.ndim)
Variable.idtype = property(lambda self: (self.ival.dtype if isinstance(self.ival, np.ndarray) else type(self.ival)))


def symbolic(fcn):
    """
    Use this to decorate a symbolic function with any return format (it will be detected automatically).
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=AnyReturnFormat, update_format=PassAnythingFormat)(fcn)


def symbolic_stateless(fcn):
    """
    Use this to decorate symbolic functions that create no state updates.  It will check that they do not change state.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=AnyReturnFormat, update_format=NoUpdatesFormat)(fcn)


def symbolic_simple(fcn):
    """
    Use this to decorate a symbolic function that takes theano tensors as inputs and returns a single tensor.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=SingleOutputFormat, update_format=PassAnythingFormat)(fcn)


def symbolic_multi(fcn):
    """
    Use this to decorate a symbolic function that takes theano tensors as inputs and returns a tuple of tensors.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=MultiOutputFormat, update_format=PassAnythingFormat)(fcn)


def symbolic_updater(fcn):
    """
    Use this to decorate a symbolic function that returns a list of updates and no outputs.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=NoOutputFormat, update_format=SomeUpdatesFormat)(fcn)


class SymbolicFunction(object):

    def __init__(self, input_format = None, output_format = None, update_format = None):

        # Cases:
        self.input_format = PassAnythingFormat if input_format is None else input_format
        self.output_format = PassAnythingFormat if output_format is None else output_format
        self.update_format = PassAnythingFormat if update_format is None else update_format

    def __call__(self, fcn):

        if inspect.isclass(fcn):
            # This is class with a __call__ method
            return _decorate_callable_class(fcn, self.input_format, self.output_format, self.update_format)

        elif hasattr(fcn, '__call__'):
            # This is a function.  It may be:
            # 1) An ordinary function
            # 2) An unbound method.
            return _SymbolicFunctionWrapper(fcn, input_format = self.input_format, output_format=self.output_format, update_format=self.update_format)

        else:
            raise Exception('Should never get here.')


def _decorate_callable_class(callable_class, input_format, output_format, update_format):

    assert hasattr(callable_class, '__call__'), "If you decorate a class with a symbolic decorator, it must "\
        "be callable.  If there's a specific method you want to decorate, decorate that instead."

    # Strategy 1: Return a new constructor that dynamically binds the function_type as a base-class when the object
    # is instantiated. (Now defunct - see git log if you want)

    # Strategy 2: Bind the function_type as a base-class to the class - the __new__ method of function_type will then be
    # called when the object is instantiated.
    class CallableSymbolicFunction(callable_class, _SymbolicFunctionWrapper):
            """
            This is a dynamic class that binds together the callable class with the symbolic function.  The idea is to make
            the callable class comply to the ISymbolicFunction interface.
            """

            # Also decorate the __call__ method, so that type checking is done.
            __call__ = _SymbolicFunctionWrapper(callable_class.__call__, input_format = input_format, output_format = output_format, update_format=update_format)

            def __init__(self, *args, **kwargs):
                _SymbolicFunctionWrapper.__init__(self, callable_class, input_format = input_format, output_format=output_format, update_format=update_format)
                callable_class.__init__(self, *args, **kwargs)

            def fcn_str(self):
                return '<%s object at %s>' % (callable_class.__name__, hex(id(self)))

    return CallableSymbolicFunction


class _SymbolicFunctionWrapper(object):
    """
    For internal use only.  Use decorators
    """

    def __init__(self, fcn, input_format, output_format, update_format, attached_instance = None):
        """
        :param fcn: The function being wrapped
        :param input_format: An IFormat object representing the input format
        :param output_format: An IFormat object representing the output format
        :param update_format: An IFormat object representing the update format.
        :param attached_instance: Will be None, unless called from __get__ (for methods)
        """
        self.fcn = fcn
        self.input_format = input_format
        self.output_format = output_format
        self.update_format = update_format
        self._dispatched_methods = {}  # Only used when fcn is an unbound method (see __get__)
        self._captured_locals = {}
        self.attached_instance = attached_instance

    def __call__(self, *args, **kwargs):
        self.input_format.check((args, kwargs), self.fcn)

        with StateCatcher(swallow_updates=False) as sc:
            if ENABLE_OMNISCENCE:
                with CaptureLocals() as c:
                    if self.attached_instance is None:
                        symbolic_return = self.fcn(*args, **kwargs)
                    else:
                        symbolic_return = self.fcn(self.attached_instance, *args, **kwargs)
                captured_anything = c.get_captured_locals()
                captured_variables = flatten_struct(captured_anything, primatives = (Variable, SharedVariable), break_into_objects=False)
                captured_locals = {k: v for k, v in captured_variables if isinstance(v, Variable)}
                self._captured_locals = captured_locals
            else:
                if self.attached_instance is None:
                    symbolic_return = self.fcn(*args, **kwargs)
                else:
                    symbolic_return = self.fcn(self.attached_instance, *args, **kwargs)
        self.update_format.check(sc.get_updates(), self.fcn)
        self.output_format.check(symbolic_return, self.fcn)
        return symbolic_return

    def scan(self, **scan_kwargs):
        """
        Apply a scan to this function.  For arguments, see thr
        :param scan_kwargs: See theano.scan doc
        :return:
        """
        outputs, updates = theano.scan(self._call_with_updates_returned, **scan_kwargs)
        for (shared_var, new_val) in updates.items():
            add_update(shared_var, new_val)
        return outputs

    def _call_with_updates_returned(self, *args, **kwargs):
        with StateCatcher(swallow_updates=True) as sc:
            outputs = self(*args, **kwargs)
        return outputs, sc.get_updates()

    def to_format(self, format_decorator):

        @format_decorator
        def conversion_wrapper(*args, **kwargs):
            formatted_args, formatted_kwargs = convert_formats((args, kwargs), src_format=conversion_wrapper.input_format, dest_format=self.input_format)
            formatted_return = self(*formatted_args, **formatted_kwargs)
            return_val = convert_formats(formatted_return, src_format = self.output_format, dest_format=conversion_wrapper.output_format)
            return return_val
        return conversion_wrapper

    def partial(self, **fixed_kwargs):
        """
        Partially define the input arguments and return a new symbolic function.
        """
        return _SymbolicFunctionWrapper(fcn=partial(self.fcn, **fixed_kwargs), input_format = PassAnythingFormat,
            output_format=self.output_format, update_format=self.update_format, attached_instance=self.attached_instance)

    def compile(self, **compilation_kwargs):
        return AutoCompilingFunction(self, **compilation_kwargs)

    def __get__(self, instance, other):
        # What's going on here:
        # self is an SymbolicFunction that wraps a method - it is created at the time the class is, before
        # any object is instantiated.  Every time the method is requested from an instantiated object, this
        # function is called.  This function has 2 jobs: 1: Make sure the dispatched method is a symbolic function
        # of the same type as this (e.g. StatelessSymbolicFunction).  2: Make sure that each time the method is
        # requested for a particular instance, we return the same method.  2 is important for (a) efficiency - there's
        # no reason to create a separate object every time we want to get the method, and (b) debugging - because we
        # attach the local variables to the method, and want to get them later, so the returned method better have
        # the same address every time we request it.
        if instance in self._dispatched_methods:
            return self._dispatched_methods[instance]
        else:
            return _SymbolicFunctionWrapper(self.fcn, input_format=self.input_format, output_format=self.output_format, update_format=self.update_format, attached_instance=instance)

    def fcn_str(self):
        if self.attached_instance is None:
            return self.fcn.__str__()
        else:
            return '%s.%s' % (self.attached_instance, self.fcn.__str__())

    def __str__(self):
        return '%s containing %s' % (self.__class__.__name__, self.fcn_str(), )

    def __repr__(self):
        return self.__str__()

    def locals(self):
        return self._captured_locals


class IFormat(object):

    @staticmethod
    def check(data, f):
        """
        Assert that data is in correct format.  Otherwise, throw SymbolicFormatError.  f is the reference to the function
        whose inputs/outputs/updates are being inspected.  f is passed in so that it can be used in the error message,
        if any.
        """


def _detect_format(data):
    if _is_tensor(data):
        return SingleOutputFormat
    elif _is_tuple_of_tensors(data):
        return MultiOutputFormat
    elif data is None:
        return NoOutputFormat
    elif _is_named_collection(data):
        return NamedCollectionFormat
    else:
        raise SymbolicFormatError("Data is not in any known format for a symbolic return: %s" % (data, ))


def convert_formats(data, src_format, dest_format):

    if src_format == dest_format:
        return data
    elif src_format is AnyReturnFormat:
        actual_src_format = _detect_format(data)
        return convert_formats(data, actual_src_format, dest_format)
    elif src_format is NoOutputFormat and dest_format is MultiOutputFormat:
        return ()
    elif src_format is SingleOutputFormat and dest_format is MultiOutputFormat:
        return (data, )
    elif src_format is MultiOutputFormat and dest_format is SingleOutputFormat:
        if len(data) > 1:
            raise SymbolicFormatError("You are trying to express multiple variables: %s in a single-variable format.  Doesn't work." % (data, ))
        return data[0]
    elif src_format is MultiOutputFormat and dest_format is NoOutputFormat:
        if len(data) > 0:
            raise SymbolicFormatError("You're trying to convert from MultiOutputFormat to NoOutputFormat, but your output tuple is not empty.  It looks like: %s" % (data, ))
    elif src_format is NamedCollectionFormat and dest_format is MultiOutputFormat:
        return tuple(data.values())
    else:
        raise SymbolicFormatError('No way to convert data from %s to %s' % (src_format, dest_format))


class PassAnythingFormat(IFormat):

    @staticmethod
    def check(data, f):
        pass


class AnyReturnFormat(IFormat):

    @staticmethod
    def check(data, f):
        try:
            _detect_format(data)  # This will check if the data is in any familiar format.
        except SymbolicFormatError:
            raise SymbolicFormatError("The return of function %s was not in any familiar format.: %s" % (f, data))


class SingleOutputFormat(IFormat):

    @staticmethod
    def check(data, f):
        if not _is_tensor(data):
            raise SymbolicFormatError('Function %s was should have returned a tensor output, but instead returned: %s' % (f, data))


class MultiOutputFormat(IFormat):

    @staticmethod
    def check(data, f):
        if not _is_tuple_of_tensors(data):
            raise SymbolicFormatError('Function %s was should have returned a tuple-of-tensors output, but instead returned: %s' % (f, data))


class NoOutputFormat(IFormat):

    @staticmethod
    def check(data, f):
        assert data is None, "Function %s should have returned no output, but it returned %s.  If your intention was to return updates, use add_update instead." % (f, data)


class NoUpdatesFormat(IFormat):

    @staticmethod
    def check(data, f):
        assert isinstance(data, list), "Updates should be in the form of a list.  Something is strange if this is not the case"
        if len(data)!=0:
            raise SymbolicFormatError("Function %s should have created no state updates, but it created updates: %s" % (f, data))


class SomeUpdatesFormat(IFormat):

    @staticmethod
    def check(data, f):
        if isinstance(data, list): "Updates should be in the form of a list.  Something is strange if this is not the case"
        if len(data) == 0:
            raise SymbolicFormatError("Function %s should have created state updates, but it failed to update any variables!" % (f, ))


class NamedCollectionFormat(IFormat):

    @staticmethod
    def check(data, f):
        if not _is_named_collection(data):
            raise SymbolicFormatError("Data should be a named collection, in a dict<string:tensor> format.  Right now it looks like this: %s" % (data, ))


class SymbolicFormatError(Exception):
    pass


def _is_tensor(arg):
    return isinstance(arg, Variable)


def _is_tuple_of_tensors(args):
    return isinstance(args, (list, tuple)) and all(isinstance(arg, Variable) for arg in args)


def _is_named_collection(arg):
    if not isinstance(arg, dict):
        return False
    if not all(isinstance(k, basestring) for k in arg.keys()):
        return False
    if not all(_is_tensor(v) for v in arg.values()):
        return False
    return True


def _get_relevant_trace_variables_and_callbacks(all_outputs_and_updates):
    """
    :param all_outputs: A list of symbolic variables returned, and update values.  This is
    :return: trace_variables, trace_callbacks
        Where:
            trace_variables is a dict<str: Variable} containing {trace_var_name: trace_var}
            trace_callbacks is a list<function> where function should do something with the named trace variable (see tdbprint for example)
    """
    if len(_TRACE_VARIABLES) == 0:
        return {}, {}

    all_leaves = set().union(*[find_leaf_ancestors(v) for v in all_outputs_and_updates])

    # Now we need to make sure the trace variables actually belong to this function.
    # The set of leaf ancestors to the trace variables should be a subset of the leaf-ancestors to the outputs/updates.
    # trace_variables = {name: var for name, var in _TRACE_VARIABLES.iteritems() if find_leaf_ancestors(var).issubset(all_leaves)}
    def computable_by_given_inputs(var, given_inputs):
        """
        Return True if the symbolic variable var depends only on the provided inputs, shared variables and constants
        """
        all_leaf_ancestors = find_leaf_ancestors(var)
        ancestors_are_computable = [(a in given_inputs) or isinstance(a, SharedVariable) or isinstance(a, tt.Constant) for a in all_leaf_ancestors]
        return all(ancestors_are_computable)

    trace_variables = {name: var for name, var in _TRACE_VARIABLES.iteritems() if computable_by_given_inputs(var, given_inputs = all_leaves)}
    # TODO: Fix.  We still have problems with accepting teave variables that don't belong.
    trace_callbacks = [_TRACE_CALLBACKS[name] for name in trace_variables if name in _TRACE_CALLBACKS]
    return trace_variables, trace_callbacks


class AutoCompilingFunction(object):
    """
    Given a Symbolic function, turn it into a compiled function that will accept and return numpy arrays.  Actual
    compilation happens on the first use of the function, since it needs to see the arguments in order to instantiate
    the input tensors. Generally you do not use this directly, instead, go:

        @symbolic
        def my_function(x):
            return x*2
        f = my_function.compile()

    f will be an AutoCompilingFunction
    """

    def __init__(self, fcn, cast_to_floatx = 'float', fixed_args = None, add_test_values = True):
        """
        :param fcn: A symbolic function (decorated with one of the above decorators)
        :param cast_to_floatx: Case inputs  to the global float type (define this in ~/.theanorc).
            'float': Just cast floats to floatX
            'all': Cast all inputs to floatX
            None: Don't cast anything to floatX
        :param fixed_args: A dict<arg_name: arg_value> of fixed arguments to the function.
        :param add_test_values: Add test values to your tensor, based on the initial value of the data provided.  Advantage
            of this is it helps you catch and locate shape errors before compiling.  Disadvantage is on large computations
            you have to do an initial pass on CPU, which can be slow.
        """
        assert isinstance(fcn, _SymbolicFunctionWrapper), 'You must pass a symbolic function.  Decorate it!'
        theano.config.compute_test_value = 'warn' if add_test_values else 'off'
        if fixed_args is not None:
            fixed_tensors = {k: (tt.constant(v) if isinstance(v, np.ndarray) else v) for k, v in fixed_args.iteritems()}
            for k, v in fixed_args.iteritems():
                if isinstance(v, (np.ndarray, Variable)):
                    fixed_tensors[k].tag.test_value = \
                        v if isinstance(v, np.ndarray) else \
                        v.get_value() if isinstance(v, SharedVariable) else \
                        v.tag.test_value if isinstance(v, Variable) else \
                        np.array(v)
            self._fcn = partial(fcn, **fixed_tensors)
        else:
            self._fcn = fcn
        self._original_fcn = fcn  # Needed for retrieveing locals hack
        self._compiled_fcn = None
        self._cast_to_floatx = cast_to_floatx
        self._local_values = None
        self._callbacks = []
        self._add_test_values = add_test_values

        # Create convenient debugging functions: showloc() and locinfo()
        __builtins__['showloc'] = show_all_locals
        __builtins__['locinfo'] = get_local_info

    def __call__(self, *args, **kwargs):
        """
        :param args, kwargs are the arguments that would go into fcn, but as real numpy arrays instead of symbols
        returns the result, in numpy arrays.
        """

        if self._compiled_fcn is None:  # Need to do first pass and compile.

            d2t = partial(_data_to_tensor, cast_to_floatx = self._cast_to_floatx, add_test_value = self._add_test_values)
            tensor_args = [d2t(arg) for arg in args]
            tensor_kwargs = OrderedDict((k, d2t(a)) for k, a in kwargs.iteritems())
            self._kwarg_order = tensor_kwargs.keys()
            args_and_kwarg_tensors = tensor_args + tensor_kwargs.values()

            with StateCatcher(swallow_updates=True) as sc:
                outputs = self._fcn(*tensor_args, **tensor_kwargs)

            updates = sc.get_updates()
            all_outputs_and_updates = convert_formats(outputs, AnyReturnFormat, MultiOutputFormat) + tuple(new for old, new in updates)
            trace_variables, trace_callbacks = _get_relevant_trace_variables_and_callbacks(all_outputs_and_updates)
            self._there_are_debug_variables = (len(trace_variables)>0 and ENABLE_TRACES) or (ENABLE_OMNISCENCE and (self._original_fcn.locals() is not None))
            self._callbacks += trace_callbacks

            if self._there_are_debug_variables:
                # Append trace variables onto output (to be stripped off later)
                self._original_output_format = _detect_format(outputs)
                if self._original_output_format is NamedCollectionFormat:
                    self._signal_names = outputs.keys()
                outputs = convert_formats(outputs, src_format=self._original_output_format, dest_format=MultiOutputFormat)
                self._trace_variable_keys = trace_variables.keys()
                self._local_variable_keys = self._original_fcn.locals().keys()
                self._n_outputs = len(outputs)
                self._n_trace_vars = len(trace_variables)
                outputs = outputs+tuple(trace_variables.values())+tuple(self._original_fcn.locals().values())

            PLATO_LOGGER.info('Compiling %s...' % (self._original_fcn.fcn_str(), ))
            self._compiled_fcn = theano.function(inputs = args_and_kwarg_tensors, outputs = outputs, updates = updates, allow_input_downcast=self._cast_to_floatx)
            PLATO_LOGGER.info('Done.\n')

        arg_and_kwarg_values = args + tuple(kwargs[k] for k in self._kwarg_order)

        # Now, run the actual numeric function!
        if self._there_are_debug_variables:
            # Separate out the debug variables from the output.
            all_out = self._compiled_fcn(*arg_and_kwarg_values)
            true_out = all_out[:self._n_outputs]
            trace_out = all_out[self._n_outputs:self._n_outputs+self._n_trace_vars]
            local_out = all_out[self._n_outputs+self._n_trace_vars:]
            trace_values = {k: v for k, v in zip(self._trace_variable_keys, trace_out)}
            _TRACE_VALUES.update(trace_values)
            self._local_values = {k: v for k, v in zip(self._local_variable_keys, local_out)}
            if self._original_output_format is NamedCollectionFormat:
                true_out = OrderedDict((k, v) for k, v in zip(self._signal_names, true_out))
            else:
                true_out = convert_formats(true_out, MultiOutputFormat, self._original_output_format)
        else:
            true_out = self._compiled_fcn(*arg_and_kwarg_values)

        for c in self._callbacks:
            c()

        return true_out

    def __str__(self):
        return 'Compiled form of %s' % (self._original_fcn.fcn_str(), )

    def add_callback(self, fcn):
        self._callbacks.append(fcn)

    def locals(self):
        return expand_struct(self._local_values)

    @property
    def symbolic(self):
        """ Return the symbolic function """
        return self._fcn


ENABLE_TRACES = True


def set_enable_traces(state):
    global ENABLE_TRACES
    ENABLE_TRACES = state


class EnableOmniscence():

    def __enter__(self):
        global ENABLE_OMNISCENCE
        ENABLE_OMNISCENCE = True

    def __exit__(self, exc_type, exc_val, exc_tb):
        global ENABLE_OMNISCENCE
        ENABLE_OMNISCENCE = False


ENABLE_OMNISCENCE = False


def set_enable_omniscence(state):
    """
    A possible useful but evil feature wherein we can peek at the local variables of a compiled function.
    """
    global ENABLE_OMNISCENCE
    ENABLE_OMNISCENCE = state


def _is_symbol_or_value(var):
    return isinstance(var, tt.TensorType) or isinstance(var, np.ndarray) or np.isscalar(var)


def _data_to_tensor(data, name = None, cast_to_floatx = True, add_test_value = True):
    """
    Given the numpy data from the first function call, create the appropriate tensors
    :param data: A numpy array, from the first call to the function.
    :param name: Optionally, a name to give the variable.
    :param cast_to_floatx: Case inputs  to the global float type (define this in ~/.theanorc).
        'float': Just cast floats to floatX
        'all': Cast all inputs to floatX
        None: Don't cast anything to floatX
    :param add_test_values: Add test values to your tensor, based on the initial value of the data provided.  Advantage
        of this is it helps you catch and locate shape errors before compiling.  Disadvantage is on large computations
        you have to do an initial pass on CPU, which can be slow.
    :return:
    """
    assert cast_to_floatx in ('float', 'all', None), 'Bad argument for cast_to_floatx: %s' % (cast_to_floatx, )
    ndim = 0 if np.isscalar(data) else data.ndim

    warn_about_floatx = False  # Too many false positives.  Got to find a better way to give this warning.

    if warn_about_floatx:
        if isinstance(data, np.ndarray) and data.dtype in (int, bool) and theano.config.floatX == 'float32':
            logging.warn("Your floatX (defined in ~/.theanorc) is float32, but you're passing in integer arrays to your function.  "
                "The problem is that most operations involving a float32 array and an int array result in a float64 array.  So what "
                "may happen is you may get a TypeError telling you that the update must have the same type as the original.  If you "
                "don't that's cool, ignore this.  Otherwise, to fix this problem, you either cast your inputs to floats beforehand, "
                "or compile your symbolic functions with: fcn.compile(cast_to_floatx='all')")

    is_dtype = lambda x, dtype: isinstance(x, dtype) or isinstance(x, np.ndarray) and x.dtype == dtype

    # Need to also downcast ints to int32 if floatX is float32, otherwise things like int_array.mean() return float64
    # objects, which (a) slows things down and (b) causes an error when you try to update 32-bit shared variabkles
    # with 64 bit values.

    dtype = \
        theano.config.floatX if (cast_to_floatx == 'all' or (cast_to_floatx=='float' and is_dtype(data, float))) else \
        'int32' if (cast_to_floatx=='float' and theano.config.floatX == 'float32' and is_dtype(data, int)) else \
        'int64' if isinstance(data, (bool, int)) else \
        'float64' if isinstance(data, float) else \
        'int8' if data.dtype==bool else \
        data.dtype
    tensor = TensorType(dtype, (None, )*ndim)(name)
    if add_test_value:
        tensor.tag.test_value = data.astype(dtype) if isinstance(data, np.ndarray) else np.array(data).astype(dtype)
    return tensor


def show_all_locals():
    locals_of_calling_frame = inspect.currentframe().f_back.f_locals
    print '=== Locals ==='
    for k, v_info in get_local_info(locals_of_calling_frame).iteritems():
        print '%s = %s' % (k, v_info)
    print '--------------'


def get_local_info(locals_of_calling_frame=None):
    if locals_of_calling_frame is None:
        locals_of_calling_frame = inspect.currentframe().f_back.f_locals
    info = {k: var_info(v) for k, v in locals_of_calling_frame.iteritems()}
    return info


def var_info(var):

    if isinstance(var, Variable) and hasattr(var.tag, 'test_value'):
        return '%s with test_value = %s' % (str(var), var_info(var.tag.test_value))
    elif isinstance(var, SharedVariable):
        return 'Shared %s value = %s' % (str(var), var_info(var.get_value()))
    elif isinstance(var, np.ndarray):
        return array_info(var)
    else:
        return str(var)


def array_info(arr):
    if arr.size <= 10:
        return '%s(%s)' % (arr.__class__.__name__, str(arr).replace('\n', ', '))
    elif arr.size <= 200000:
        return '%s of shape %s in %s<=arr<=%s' % (arr.__class__.__name__, arr.shape, np.min(arr), np.max(arr))
    else:
        return '%s of shape %s' % (arr.__class__.__name__, arr.shape, )


def find_shared_ancestors(variable):
    """
    Given a variable, return a list of all shared variables that it depends upon.  This can be useful for
    finding the parameters to update when trying to optimize this variable in some way.
    :param variable: A theano variable
    :return: A list of SharedVariables.
    """
    if isinstance(variable, SharedVariable):
        return [variable]
    else:
        return list(set(sum([find_shared_ancestors(p) for p in variable.get_parents()], [])))


def find_all_ancestors(variable, memo = None):
    """
    Return a set including the all ancestors of the given variable
    :param variable: A Theano Tensor
    :return: A set containing all ancestors, including the given variable.
    """

    if memo is None:
        memo = set()

    memo.add(variable)

    for p in variable.get_parents():
        if p not in memo:
            find_all_ancestors(p, memo = memo)

    return memo


def find_leaf_ancestors(variable):
    all_ancestors = find_all_ancestors(variable)
    leaf_ancestors = {var for var in all_ancestors if len(var.get_parents()) == 0}
    return leaf_ancestors


_TRACE_VARIABLES = OrderedDict()  # A dict of trace-variable-name: Trace Variable
_TRACE_VALUES = OrderedDict()  # A dict of trace variable name: Most recently computed value
_TRACE_CALLBACKS = OrderedDict()  # A dict of trace-variable-name: Callback to call after trace ver is used.


def get_tdb_traces():
    return _TRACE_VALUES


def tdb_trace(var, name = None, callback = None):
    if name is None:
        # TODO: Get default by sneakily grabbing name from calling scope.
        name = '%s@%s' % (str(var), hex(id(var)))
    _TRACE_VARIABLES[name] = var
    if callback is not None:
        _TRACE_CALLBACKS[name] = callback


def clear_tdb_traces():
    _TRACE_CALLBACKS.clear()
    _TRACE_VARIABLES.clear()


def printit(var_name, var_val):
    print '%s: %s' % (var_name, var_val)


def tdbprint(var, name = None):
    if name is None:
        # TODO: Get default by sneakily grabbing name from calling scope.
        name = '%s@%s' % (str(var), hex(id(var)))
    tdb_trace(var, name, callback = lambda: printit(var_name = name, var_val = _TRACE_VALUES[name]))


STATE_CATCHER = None


def _get_state_catcher():
    return STATE_CATCHER


def _set_state_catcher(val):
    global STATE_CATCHER
    STATE_CATCHER = val


def add_update(shared_var, new_val):
    """
    :param shared_var: A theano SharedVariable object
    :param new_val: The new value for this sharedvariable to take on (usually a TensorVariable)
    """
    assert isinstance(shared_var, SharedVariable), 'shared_var must be a theano shared variable.'
    state_catcher = _get_state_catcher()
    assert state_catcher is not None, "You tried to add an update from a function that is not symbolic, and is not being called by a symbolic function."
    state_catcher.add_update(shared_var, new_val)


class StateCatcher(object):
    """
    Used to catch updates.  Usage:

    with StateCatcher() as sc:
        # Code here
    updates = sc.get_updates()  # A List<Tuple<SharedVariable, Variable>> contaning all updates in which add_update was called.
    """

    def __init__(self, swallow_updates = False):
        """
        :param swallow_updates: A boolean.  True if you'd like to "swallow" all updates produced, and not pass them on to any
            outer state-catcher.  False if you'd like to pass updates to the outer StateCatcher.
        :return:
        """
        self.swallow_updates = swallow_updates

    def __enter__(self):
        self._outer_catcher = _get_state_catcher()
        _set_state_catcher(self)
        self._updates = OrderedDict()
        return self

    def __exit__(self, *args):
        _set_state_catcher(self._outer_catcher)

    def add_update(self, shared_var, new_val):
        assert shared_var not in self._updates, "You tried to update shared-variable %s with tensor %s, but you've already updated it with tensor %s" % (shared_var, new_val, self._updates[shared_var])
        self._updates[shared_var] = new_val
        if self._outer_catcher is not None and not self.swallow_updates:  # Allows for nested StateCatchers (outer ones do not have to worry about inner ones stealing their updates)
            self._outer_catcher.add_update(shared_var, new_val)

    def get_updates(self):
        return self._updates.items()


def assert_compatible_shape(actual_shape, desired_shape, name = None):
    """
    Return a boolean indicating whether the actual shape is compatible with the desired shape.  "None" serves as a wildcard.
    :param actual_shape: A tuple<int>
    :param desired_shape: A tuple<int> or None
    :return: A boolean

    examples (actual_desired, desired_shape : result)
    (1, 2), None: True        # Because None matches everything
    (1, 2), (1, None): True   # Because they have the same length, 1 matches 1 and 2 matches None
    (1, 2), (1, 3): False     # Because 2 != 3
    (1, 2, 1), (1, 2): False  # Because they have different lengths.
    """
    return desired_shape is None or len(actual_shape) == len(desired_shape) and all(ds is None or s==ds for s, ds in zip(actual_shape, desired_shape)), \
        "Actual shape %s%s did not correspond to specified shape, %s" % (actual_shape, '' if name is None else ' of %s' %(name, ), desired_shape)


def initialize_param(initial_value, shape = None, name = None, cast_floats_to_floatX = True):
    """
    Takes care of the common stuff associated with initializing a parameter.  There are a few ways you may want to
    instantiate a parameter:
    - With a numpy array, in which case you'll want to make sure it's the appropriate shape.
    - With a scalar, in which case you just want a scalar shared variable.
    - With a scalar and a shape, in which case you want an array of that shape filled with the value of the scalar.
    - With a symbolic variable descenting from some other shared variable - this is the case when you want to tie
      parameters together, or make the bias be the result of a previous computation, etc.
    - With a function and shape, in which case the function should return an initial numpy array given the shape.
    - With None, which we take to mean that this was an optional variable that should not be included, so return None
      for variable, param, and shape.

    :param initial_value: An array, scalar, or symbolic variable.:
    :param shape: The shape that the variable should have.  None if it is already fully specified by initial_value.
        If shape is a tuple, elements of shape s can be:
        - integers: In which case, they mean (the dimension of in this direction shall be <s>
        - None: In which case, the initial_value must be defined as an array, and the array may have any shape along this axis.
    :param name: Optionally, the name for the shared variable.
    :return: (variable, param): Variable is the shared variable, and param is the associated parameter.  If you
        instantiated with scalar or ndarray, variable and param will be the same object.
    """

    if isinstance(shape, int):
        shape = (shape, )

    typecast = lambda x: x.astype(theano.config.floatX) if cast_floats_to_floatX and x.dtype in ('float', 'float64', 'float32') else x

    if np.isscalar(initial_value):
        if shape is None:
            initial_value = np.array(initial_value)
        else:
            initial_value = np.zeros(shape)+initial_value
        initial_value = typecast(initial_value)
    elif hasattr(initial_value, '__call__'):
        assert shape is not None, "If you initialize with a function, you must provide a shape."
        initial_value = initial_value(shape)

    if isinstance(initial_value, np.ndarray):
        assert_compatible_shape(initial_value.shape, shape, name = name)
        variable = theano.shared(typecast(initial_value), name = name, borrow = True, allow_downcast=True)
        params = [variable]
        variable_shape = initial_value.shape
    elif isinstance(initial_value, Variable):
        assert name is None or initial_value.name == name, "Can't give name '%s' to an already-existing symbolic variable" % (name, )
        params = find_shared_ancestors(initial_value)
        # Note to self: possibly remove this constraint for things like factored weight matrices?
        if len(params)==1 and initial_value is params[0]:
            variable_shape = initial_value.get_value().shape
            assert_compatible_shape(variable_shape, shape, name = name)
        else:
            raise NotImplementedError("Can't yet get variable shape from base-params, though this can be done cheaply in "
                'Theano by compiling a function wholse input is the params and whose output is the shape.')
        variable = initial_value
    elif initial_value is None:
        variable = None
        params = []
        variable_shape = None
    else:
        raise Exception("Don't know how to instantiate variable from %s" % initial_value)
    return variable, params, variable_shape


def create_shared_variable(initializer_fcn, shape = None, name = None, cast_floats_to_floatX = True):
    """
    :param initializer_fcn: Can be:
        - An array.  It may be cast to floatX.  It's verified with shape if shape is provided
        - A function which takes the shape and turns it into the array.
        - A scalar, in which case it's broadcase over shape.
    :param shape: Either a tuple or an integer
    :return: A shared variable, containing the numpy array returned by the initializer.
    """
    shared_var, _, _ = initialize_param(initializer_fcn, shape = shape, name = name, cast_floats_to_floatX=cast_floats_to_floatX)
    return shared_var
