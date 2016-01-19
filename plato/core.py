from collections import OrderedDict
from functools import partial
import inspect
import logging
from general.local_capture import CaptureLocals
from general.nested_structures import flatten_struct, expand_struct
from theano.compile.sharedvalue import SharedVariable
from theano.gof.graph import Variable
import theano.tensor as tt
from theano.tensor.type import TensorType
import theano
import numpy as np
from theano.updates import OrderedUpdates

"""
It is better not to look at the things happening in here.  It's beautiful on the outside but not on the inside.
All you need to know is this:

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


# Add properties to the "Variable" class (the base class of all symbolic variables), so that you easily inspect
# the initial values that are attached to them.
Variable.ival = property(lambda self: (self.get_value() if isinstance(self, SharedVariable) else self.tag.test_value))
Variable.ishape = property(lambda self: self.ival.shape)


def symbolic(fcn):
    """
    Use this to decorate a symbolic function with any return format (it will be detected automatically).
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=AnyReturnFormat)(fcn)


def symbolic_simple(fcn):
    """
    Use this to decorate a symbolic function that takes theano tensors as inputs and returns a single tensor.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=SingleOutputFormat)(fcn)


def symbolic_multi(fcn):
    """
    Use this to decorate a symbolic function that takes theano tensors as inputs and returns a tuple of tensors.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=MultiOutputFormat)(fcn)


def symbolic_single_output_updater(fcn):
    """
    Use this to decorate a symbolic function that takes theano tensors as inputs and returns a single tensor and a list of updates.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=SingleOutputUpdater)(fcn)


def symbolic_updater(fcn):
    """
    Use this to decorate a symbolic function that returns a list of updates.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=UpdateFormat)(fcn)


def symbolic_standard(fcn):
    """
    Use this to decorate a symbolic function that returns a tuple of outputs and a list of updates.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=StandardFormat)(fcn)


class SymbolicFunction(object):

    def __init__(self, input_format = None, output_format = None):

        # Cases:
        # 1) Ordinary function
        self.input_format = input_format
        self.output_format = output_format

    def __call__(self, fcn):

        if inspect.isclass(fcn):
            # This is class with a __call__ method
            return _decorate_callable_class(fcn, self.input_format, self.output_format)

        elif hasattr(fcn, '__call__'):
            # This is a function.  It may be:
            # 1) An ordinary function
            # 2) An unbound method.
            return SymbolicFunctionWrapper(fcn, input_format = self.input_format, output_format=self.output_format)

        else:
            raise Exception('Should never get here.')


def _decorate_callable_class(callable_class, input_format, output_format):

    assert hasattr(callable_class, '__call__'), "If you decorate a class with a symbolic decorator, it must "\
        "be callable.  If there's a specific method you want to decorate, decorate that instead."

    # Strategy 1: Return a new constructor that dynamically binds the function_type as a base-class when the object
    # is instantiated. (Now defunct - see git log if you want)

    # Strategy 2: Bind the function_type as a base-class to the class - the __new__ method of function_type will then be
    # called when the object is instantiated.
    class CallableSymbolicFunction(callable_class, SymbolicFunctionWrapper):
            """
            This is a dynamic class that binds together the callable class with the symbolic function.  The idea is to make
            the callable class comply to the ISymbolicFunction interface.
            """

            # Also decorate the __call__ method, so that type checking is done.
            __call__ = SymbolicFunctionWrapper(callable_class.__call__, input_format = input_format, output_format = output_format)

            def __init__(self, *args, **kwargs):
                SymbolicFunctionWrapper.__init__(self, callable_class, input_format = input_format, output_format=output_format)
                callable_class.__init__(self, *args, **kwargs)

            def fcn_str(self):
                return '<%s object at %s>' % (callable_class.__name__, hex(id(self)))

    return CallableSymbolicFunction


class SymbolicFunctionWrapper(object):
    """
    For internal use only.  Use decorators
    """

    def __init__(self, fcn, input_format = None, output_format = None, attached_instance = None):
        """
        :param fcn: The function being wrapped
        :param input_format: An IFormat object representing the input format
        :param output_format: An IFormat object representing the output format
        :param attached_instance: Will be None, unless called from __get__ (for methods)
        """
        self.fcn = fcn
        self.input_format = input_format
        self.output_format = output_format
        self._dispatched_methods = {}  # Only used when fcn is an unbound method (see __get__)
        self._captured_locals = {}
        self.attached_instance = attached_instance

    def __call__(self, *args, **kwargs):
        self.input_format.check((args, kwargs))

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
        self.output_format.check(symbolic_return)
        return symbolic_return

    def to_format(self, format_decorator):

        @format_decorator
        def conversion_wrapper(*args, **kwargs):
            formatted_args, formatted_kwargs = convert_formats((args, kwargs), src_format=conversion_wrapper.input_format, dest_format=self.input_format)
            formatted_return = self(*formatted_args, **formatted_kwargs)
            return_val = convert_formats(formatted_return, src_format = self.output_format, dest_format=conversion_wrapper.output_format)
            return return_val
        return conversion_wrapper

    def partial(self, **fixed_kwargs):
        raise NotImplementedError('Future-plan: Allow sequential narrowing of args.')

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
            return SymbolicFunctionWrapper(self.fcn, input_format=self.input_format, output_format=self.output_format, attached_instance=instance)

    def fcn_str(self):
        if self.attached_instance is None:
            return self.fcn.__str__()
        else:
            return '%s.%s' % (self.attached_instance.__str__(), self.fcn.__str__())

    def __str__(self):
        return '%s containing %s' % (self.__class__.__name__, self.fcn_str(), )

    def locals(self):
        return self._captured_locals


class IFormat(object):

    @staticmethod
    def check(data):
        """ Assert that data is in correct format.  Otherwise, throw SymbolicFormatError """


def convert_formats(data, src_format, dest_format):

    if src_format == dest_format:
        return data
    elif src_format is AnyReturnFormat and dest_format is StandardFormat:
        return detect_return_value(data, return_outputs_in_tuple=True)
    elif src_format is SingleOutputFormat and dest_format is StandardFormat:
        return (data, ), []
    elif src_format is UpdateFormat and dest_format is StandardFormat:
        return (), data
    elif src_format is StandardFormat and dest_format is SingleOutputFormat:
        outputs, updates = data
        assert len(updates) == 0, 'Cannot convert to single-return format if there are state updates.'
        assert len(outputs) == 1, "Can only convert to single-return format if there's a single return value.  Got %s" % (len(outputs), )
        return outputs[0]
    elif src_format is StandardFormat and dest_format is MultiOutputFormat:
        outputs, updates = data
        assert len(updates) == 0, 'Cannot convert to multi-return format if there are state updates.'
        return outputs
    elif src_format is SingleOutputFormat and dest_format is SingleOutputUpdater:
        output = data
        return output, []
    else:
        raise SymbolicFormatError('No way to convert data from %s to %s' % (src_format, dest_format))


class PassAnythingFormat(IFormat):

    @staticmethod
    def check(data):
        pass


class AnyReturnFormat(IFormat):

    @staticmethod
    def check(data):
        detect_return_value(data)  # This will check if the data is in any familiar format.


class StandardFormat(IFormat):

    @staticmethod
    def check(data):
        if isinstance(data, SymbolicReturn):
            # Type checked already.
            return
        if not (isinstance(data, tuple) and len(data)==2):
            raise SymbolicFormatError('You did not return a 2-tuple of outputs, updates.  You returned %s' % (data, ))
        outputs, updates = data
        MultiOutputFormat.check(outputs)
        UpdateFormat.check(updates)


class SingleOutputFormat(IFormat):

    @staticmethod
    def check(data):
        if not _is_tensor(data):
            raise SymbolicFormatError('You did not return a tensor output.  You returned: %s' % (data, ))


class SingleOutputUpdater(IFormat):

    @staticmethod
    def check(data):
        if not (isinstance(data, tuple) and len(data)==2):
            raise SymbolicFormatError('You did not return a 2-tuple of outputs, updates.  You returned %s' % (data, ))
        outputs, updates = data
        SingleOutputFormat.check(outputs)
        UpdateFormat.check(updates)


class MultiOutputFormat(IFormat):

    @staticmethod
    def check(data):
        if not _is_tuple_of_tensors(data):
            raise SymbolicFormatError('You did not return a tuple of outputs.  You returned: %s' % (data, ))


class UpdateFormat(IFormat):

    @staticmethod
    def check(data):
        if not _is_updates_list(data):
            raise SymbolicFormatError('Updates were not in the format of a list of 2-tuples [(shared_0, new_val_0), (shared_1, new_val_1), ...].'
                '\nThey were returned as: %s' % (data, ))


class SymbolicFormatError(Exception):
    pass


def _is_tensor(arg):
    return isinstance(arg, Variable)


def _is_tuple_of_tensors(args):
    return isinstance(args, (list, tuple)) and all(isinstance(arg, Variable) for arg in args)


def _is_updates_list(updates):
    """
    Return True if updates is a proper list of updates and False if not.
    :return:
    """
    return (isinstance(updates, OrderedUpdates) or (isinstance(updates, list) and all(isinstance(up, tuple) and len(up)==2 for up in updates) and
        all(isinstance(old, SharedVariable) and isinstance(new, Variable) for old, new in updates)))


def detect_return_value(return_info, return_outputs_in_tuple = False):
    """
    :param return_info: Whatever is returned from a symbolic function.
    :return: In one of two formats, depending on whether output is returned as a single or not.
        output, [(shared_0, new_val_0), ...]
        (output_0, ...), [(shared_0, new_val_0), ...]
    """
    if isinstance(return_info, tuple) and len(return_info)==2 and (_is_tensor(return_info[0]) or _is_tuple_of_tensors(return_info[0])) and _is_updates_list(return_info[1]):
        outputs, updates = return_info
    elif isinstance(return_info, SymbolicReturn):
        outputs, updates = return_info
    elif _is_updates_list(return_info):
        outputs = ()
        updates = return_info
    elif _is_tensor(return_info) or _is_tuple_of_tensors(return_info):
        outputs = return_info
        updates = []
    else:
        raise SymbolicFormatError('Return value was not in any known format: %s' % (return_info, ))

    if return_outputs_in_tuple and _is_tensor(outputs):
        outputs = (outputs, )

    if isinstance(updates, OrderedUpdates):
        updates = [(k, v) for k, v in updates.iteritems()]

    return outputs, updates


def _list_all_output_variables(return_info):
    outputs, updates = detect_return_value(return_info, return_outputs_in_tuple=True)
    out_and_up = outputs + tuple(new for old, new in updates)
    return out_and_up


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

    def __init__(self, fcn, cast_to_floatx = 'float', fixed_args = None):
        """
        :param fcn: A symbolic function (decorated with one of the above decorators)
        :param cast_to_floatx: Case inputs  to the global float type (define this in ~/.theanorc).
            'float': Just cast floats to floatX
            'all': Cast all inputs to floatX
            None: Don't cast anything to floatX
        :param fixed_args: A dict<arg_name: arg_value> of fixed arguments to the function.
        :return:
        """
        assert isinstance(fcn, SymbolicFunctionWrapper), 'You must pass a symbolic function.  Decorate it!'

        self._fcn = fcn if fixed_args is None else partial(fcn, **{k: (tt.constant(v) if isinstance(v, np.ndarray) else v) for k, v in fixed_args.iteritems()})
        self._original_fcn = fcn  # Needed for retrieveing locals hack
        self._compiled_fcn = None
        self._cast_to_floatx = cast_to_floatx
        self._local_values = None
        self._callbacks = []

        # Create convenient debugging functions: showloc() and locinfo()
        theano.config.compute_test_value = 'warn'
        __builtins__['showloc'] = show_all_locals
        __builtins__['locinfo'] = get_local_info

    def __call__(self, *args, **kwargs):
        """
        :param args, kwargs are the arguments that would go into fcn, but as real numpy arrays instead of symbols
        returns the result, in numpy arrays.
        """

        if self._compiled_fcn is None:

            d2t = partial(_data_to_tensor, cast_to_floatx = self._cast_to_floatx, test = True)
            tensor_args = [d2t(arg) for arg in args]
            tensor_kwargs = OrderedDict((k, d2t(a)) for k, a in kwargs.iteritems())
            self._kwarg_order = tensor_kwargs.keys()
            args_and_kwarg_tensors = tensor_args + tensor_kwargs.values()
            return_value = self._fcn(*tensor_args, **tensor_kwargs)

            outputs, updates = detect_return_value(return_value)
            all_outputs_and_updates = _list_all_output_variables(return_value)
            trace_variables, trace_callbacks = _get_relevant_trace_variables_and_callbacks(all_outputs_and_updates)
            self._there_are_debug_variables = (len(trace_variables)>0 and ENABLE_TRACES) or (ENABLE_OMNISCENCE and (self._original_fcn.locals() is not None))
            self._callbacks += trace_callbacks

            if self._there_are_debug_variables:
                # Append trace variables onto output (to be stripped off later)
                self._single_output = _is_tensor(outputs)
                if self._single_output:
                    outputs = (outputs, )
                self._trace_variable_keys = trace_variables.keys()
                self._local_variable_keys = self._original_fcn.locals().keys()
                self._n_outputs = len(outputs)
                self._n_trace_vars = len(trace_variables)
                outputs = outputs+tuple(trace_variables.values())+tuple(self._original_fcn.locals().values())

            self._compiled_fcn = theano.function(inputs = args_and_kwarg_tensors, outputs = outputs, updates = updates, allow_input_downcast=self._cast_to_floatx)

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

            # numeric_output = all_out[:-len(self._trace_variable_keys)]
            if self._single_output:
                true_out, = true_out
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


class EnableOmbniscence():

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


def _data_to_tensor(data, name = None, cast_to_floatx = True, test = True):
    """
    Given the numpy data from the first function call, create the appropriate tensors
    :param data: A numpy array, from the first call to the function.
    :param name: Optionally, a name to give the variable.
    :param cast_to_floatx: Case inputs  to the global float type (define this in ~/.theanorc).
        'float': Just cast floats to floatX
        'all': Cast all inputs to floatX
        None: Don't cast anything to floatX
    :param test:
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
    if test:
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


class SymbolicReturn(object):

    def __init__(self, outputs = (), updates = []):
        if not (isinstance(outputs, (list, tuple)) and all(isinstance(out, Variable) for out in outputs)):
            raise SymbolicFormatError('Outputs must a tuple of tensors.  They were %s instead' % (outputs, ))
        if not (isinstance(updates, list) and all(len(up)==2 for up in updates) and
                all(isinstance(old, SharedVariable) and isinstance(new, Variable) for old, new in updates)):
            raise SymbolicFormatError('Updates must be a list of 2-tuples of (shared_variable, update_tensor).  We got %s instead' % (updates, ))
        self.outputs = tuple(outputs) if not isinstance(outputs, tuple) else outputs
        self.updates = updates

    def __iter__(self):
        return (self.outputs, self.updates).__iter__()


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
