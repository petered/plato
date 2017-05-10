from collections import OrderedDict
from contextlib import contextmanager
from functools import partial
import inspect
import logging
from artemis.general.local_capture import CaptureLocals
from artemis.general.nested_structures import flatten_struct, expand_struct, NestedType
from artemis.general.should_be_builtins import izip_equal
from scipy.sparse.csr import csr_matrix
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
logging.basicConfig()
PLATO_LOGGER = logging.getLogger('plato')
PLATO_LOGGER.setLevel(logging.INFO)

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


def symbolic_named_output(fcn):
    """
    Use this to decorate a symbolic function that returns a list of updates and no outputs.
    """
    return SymbolicFunction(input_format=PassAnythingFormat, output_format=NamedCollectionFormat, update_format=PassAnythingFormat)(fcn)


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

        with CaptureUpdates(swallow=False) as sc:
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
        :param scan_kwargs: See theano.scan doc: http://deeplearning.net/software/theano/library/scan.html#theano.scan
            Summary, inputs are taken in the order:
            [sequences[0], ... sequences[-1], outputs_info[0], ... outputs_info[-1], non_sequences[0], ... non_sequences[-1]]
        :return:
        """
        outputs, updates = theano.scan(self._call_with_updates_returned, **scan_kwargs)

        if self._had_to_add_dummies:
            # See why this is necessary: https://groups.google.com/forum/#!topic/theano-users/F0-EeC0Lsl8
            # Basically, we need to undo some evil that is done in theano's scan function.  See _call_with_updates_returned
            outputs = outputs[:-2]

        if len(self._trace_info)>0:
            trace_outputs = outputs[-len(self._trace_info):]
            outputs = outputs[:-len(self._trace_info)]
            for (trace_name, (_, batch_in_scan, callback)), trace_output in izip_equal(self._trace_info.iteritems(), trace_outputs):
                CaptureTraceVariables.CURRENT_CATCHER.add_trace(variable=trace_output if batch_in_scan else trace_output[-1], name=trace_name, batch_in_scan=batch_in_scan, callback=callback)

        if self._single_output and isinstance(outputs, (list, tuple)):
            assert len(outputs)==1, 'This should always be true, and you should call Peter if it is not.  +3163004422 seven'
            outputs, = outputs
        for (shared_var, new_val) in updates.items():
            add_update(shared_var, new_val)
        return outputs

    def eval(self, *args, **kwargs):
        """
        Compile and evaluate the function for the given inputs.
        :param args: Arguments to the function
        :param kwargs: Keyword
        :return:
        """
        f = self.compile()
        return f(*args, **kwargs)

    def __eq__(self, other):
        # Note: This is a bit of a lazy implementation - just tests if the wrapped functions are the same.  Not sure what
        # contexts this will be used in so it may be necessary to wrap other attributes.
        if isinstance(other, self.__class__):
            if self.fcn==other.fcn:
                return True
        return False

    def _call_with_updates_returned(self, *args, **kwargs):
        with CaptureUpdates(swallow=True) as sc, CaptureTraceVariables(swallow=True) as traces:
            outputs = self(*args, **kwargs)

        self._single_output = isinstance(outputs, Variable)
        self._trace_info = traces.get_trace_variable_info()

        if self._single_output and len(traces)>0:
            outputs = (outputs, )
        elif outputs is None:
            outputs = (tt.zeros(), )

        if len(traces)>0:
            outputs = outputs + tuple(traces.values())

        self._had_to_add_dummies = isinstance(outputs, (list, tuple)) and len(outputs)==1 # Necessary evil to force theano.scan to return collection even if length is 1.
        if self._had_to_add_dummies:
            outputs = outputs + type(outputs)([tt.zeros(()), tt.zeros(())])

        return outputs, OrderedDict(sc.get_updates())

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
        # fixed_kwargs = {k: (tt.constant(v) if isinstance(v, np.ndarray) else v) for k, v in fixed_kwargs.iteritems()}  # This prevents
        fixed_kwargs = {k: (tt.constant(v) if isinstance(v, np.ndarray) else v) for k, v in fixed_kwargs.iteritems()}  # This prevents
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
    if data is None:
        return NoOutputFormat
    elif _is_tensor(data):
        return SingleOutputFormat
    elif _is_tuple_of_tensors(data):
        return MultiOutputFormat
    elif _is_tuple_of_tuples_of_tensors(data):
        return CollectionOfCollectionsOfTensorsFormat
    elif _is_named_collection(data):
        return NamedCollectionFormat
    elif _is_constant(data):
        return ConstantFormat
    else:
        raise SymbolicFormatError("Data is not in any known format for a symbolic return: %s" % (data, ))


def convert_formats(data, src_format, dest_format):

    if src_format == dest_format:
        if src_format == MultiOutputFormat and isinstance(data, list):
            data = tuple(data)
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
        pass
        # try:
        #     _detect_format(data)  # This will check if the data is in any familiar format.
        # except SymbolicFormatError:
        #     raise SymbolicFormatError("The return of function %s was not in any familiar format.: %s" % (f, data))


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


class CollectionOfCollectionsOfTensorsFormat(IFormat):

    @staticmethod
    def check(data, f):
        if not _is_tuple_of_tuples_of_tensors(data):
            raise SymbolicFormatError("Data should be a collection of collections of tensors.  Right now it looks like this: %s" % (data, ))


class ConstantFormat(IFormat):

    @staticmethod
    def check(data, f):
        if not isinstance(data, (float, int, np.ndarray)):
            raise SymbolicFormatError("Data should be a constant, numeric data (numpy or python float, etc).  Right now it looks like this: %s" % (data, ))


class SymbolicFormatError(Exception):
    pass



def _is_tensor(arg):
    return isinstance(arg, (Variable, np.ndarray))


def _is_tuple_of_tensors(args):
    return isinstance(args, (list, tuple)) and all(_is_tensor(arg) for arg in args)


def _is_tuple_of_tuples_of_tensors(args):
    return isinstance(args, (list, tuple)) and all(_is_tuple_of_tensors(a) for a in args)


def _is_named_collection(arg):
    if not isinstance(arg, dict):
        return False
    if not all(isinstance(k, (basestring, int)) for k in arg.keys()):
        return False
    if not all(_is_tensor(v) for v in arg.values()):
        return False
    return True


def _is_constant(arg):
    return isinstance(arg, (float, int, np.ndarray, np.number))


# def _get_relevant_trace_variables_and_callbacks(all_outputs_and_updates):
#     """
#     :param all_outputs: A list of symbolic variables returned, and update values.  This is
#     :return: trace_variables, trace_callbacks
#         Where:
#             trace_variables is a dict<str: Variable} containing {trace_var_name: trace_var}
#             trace_callbacks is a list<function> where function should do something with the named trace variable (see tdbprint for example)
#     """
#     # TODO: Delete: This function has lost its purpose in life (CaptureTraceVariables superseded it).  But the code may be relevant still,
#     # so for now it lives in the grey, to maybe be revived some day.
#     if len(_TRACE_VARIABLES) == 0:
#         return {}, {}
#
#
#
#     all_leaves = set().union(*[find_leaf_ancestors(v) for v in all_outputs_and_updates])
#
#     # Now we need to make sure the trace variables actually belong to this function.
#     # The set of leaf ancestors to the trace variables should be a subset of the leaf-ancestors to the outputs/updates.
#     # trace_variables = {name: var for name, var in _TRACE_VARIABLES.iteritems() if find_leaf_ancestors(var).issubset(all_leaves)}
#     def computable_by_given_inputs(var, given_inputs):
#         """
#         Return True if the symbolic variable var depends only on the provided inputs, shared variables and constants
#         """
#         all_leaf_ancestors = find_leaf_ancestors(var)
#         ancestors_are_computable = [(a in given_inputs) or isinstance(a, SharedVariable) or isinstance(a, tt.Constant) for a in all_leaf_ancestors]
#         return all(ancestors_are_computable)
#
#     trace_variables = OrderedDict((name, var) for name, var in _TRACE_VARIABLES.iteritems() if computable_by_given_inputs(var, given_inputs = all_leaves))
#     # TODO: Fix.  We still have problems with accepting leaf variables that don't belong.
#     trace_callbacks = [_TRACE_CALLBACKS[name] for name in trace_variables if name in _TRACE_CALLBACKS]
#     return trace_variables, trace_callbacks


def flatten_tensor_struct(tensor_struct):
    flat_struct = []
    for t in tensor_struct:
        if isinstance(t, (list, tuple)):
            flat_struct += flatten_tensor_struct(t)
        else:
            flat_struct.append(t)
    return flat_struct


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

    def __init__(self, fcn, cast_to_floatx = 'float', fixed_args = None, add_test_values = False, debug_print_shapes=False, resettable=False, **theano_function_kwargs):
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
        self._debug_print_shapes = debug_print_shapes
        self.theano_function_kwargs = theano_function_kwargs
        self.resettable = resettable
        self._input_format = None
        self._output_format = None
        self.updated_variables = None  # Used in reset()

        # Create convenient debugging functions: showloc() and locinfo()
        __builtins__['showloc'] = show_all_locals
        __builtins__['locinfo'] = get_local_info

    def __call__(self, *args, **kwargs):
        """
        :param args, kwargs are the arguments that would go into fcn, but as real numpy arrays instead of symbols
        returns the result, in numpy arrays.
        """
        # Remove shared variables.
        # args = tuple(a for a in args if not isinstance(a, SharedVariable))
        # kwargs = {k: v for k, v in kwargs.iteritems() if isinstance(v, SharedVariable)}

        input_data = (args, kwargs)

        if self._compiled_fcn is None:  # Runs on the first pass

            # Find tensor versions of inputs based on data in first-call, collect list of inputs
            self._input_format = NestedType.from_data(input_data)
            flat_input_data = self._input_format.get_leaves(input_data)
            args_and_kwarg_tensors = [_data_to_tensor(d, cast_to_floatx = self._cast_to_floatx, add_test_value = True if self._add_test_values else 'shape') for d in flat_input_data]

            # assert not any(isinstance(v, SharedVariable) for v in args_and_kwarg_tensors), "You can't pass SharedVariables into a compiled function, because this causes chaos when a different shard variabls is passed in."
            self._shared_var_inputs = [trace_value for trace_value in args_and_kwarg_tensors if isinstance(trace_value, SharedVariable)]

            tensor_args, tensor_kwargs = self._input_format.expand_from_leaves(args_and_kwarg_tensors, check_types=False)  # Because types will be different

            # Call the function to get symbolic outputs and updates
            PLATO_LOGGER.info('Running first pass of function {f} with test values {test_state}...'.format(f=self._original_fcn.fcn_str(), test_state = 'on' if self._add_test_values else 'off'))
            with CaptureUpdates(swallow=True) as sc, CaptureTraceVariables(swallow=True) as traces:
                outputs = self._fcn(*tensor_args, **tensor_kwargs)
            if outputs is None:
                outputs = ()
            PLATO_LOGGER.info('Done.')
            updates = sc.get_updates()

            # Detect output format, collect list of outputs
            self._output_format = NestedType.from_data(outputs)
            flat_output_tensors = self._output_format.get_leaves(outputs) if outputs is not None else []

            # If necessary, save update info for debug print
            if self._debug_print_shapes:
                self._original_updates = updates
                self._old_update_shapes = [old.get_value().shape for old, new in updates]

            # Find and add any trace variables that may have been added (with tdb_trace, tdbplot, etc) to the list of outputs
            all_outputs_and_updates = self._output_format.get_leaves(outputs) + [new for old, new in updates]
            # trace_variables, trace_callbacks = _get_relevant_trace_variables_and_callbacks(all_outputs_and_updates)
            # self._there_are_debug_variables = (len(trace_variables)>0 and ENABLE_TRACES) or (ENABLE_OMNISCENCE and (self._original_fcn.locals() is not None))

            self._there_are_debug_variables = (len(traces) > 0 and ENABLE_TRACES) or (ENABLE_OMNISCENCE and (self._original_fcn.locals() is not None))

            self._callbacks += traces.get_callbacks()
            if self._there_are_debug_variables:
                # Append trace variables onto output (to be stripped off later)
                # outputs = convert_formats(outputs, src_format=self._original_output_format, dest_format=MultiOutputFormat)

                self._trace_variable_keys = traces.keys()
                # self._trace_variable_keys = trace_variables.keys()
                self._local_variable_keys = self._original_fcn.locals().keys()
                self._n_outputs = len(flat_output_tensors)
                self._n_trace_vars = len(traces)
                flat_output_tensors = flat_output_tensors+traces.values()+self._original_fcn.locals().values()

            # Compile the theano function
            PLATO_LOGGER.info('Compiling %s with %s inputs, %s outputs, %s updates' % (self._original_fcn.fcn_str(), len(args_and_kwarg_tensors), 1 if isinstance(outputs, Variable) else 0 if outputs is None else len(outputs), len(updates)))
            args_and_kwarg_tensors = [a for a in args_and_kwarg_tensors if not isinstance(a, SharedVariable)]  # Remove shared variables from passed-in tensor args
            if self.resettable:
                self.updated_variables = [shared_var for shared_var, update in updates]
                self._original_variable_values = [var.get_value() for var in self.updated_variables]
            self._compiled_fcn = theano.function(inputs = args_and_kwarg_tensors, outputs = flat_output_tensors, updates = updates, allow_input_downcast=self._cast_to_floatx, **self.theano_function_kwargs)
            PLATO_LOGGER.info('Done.')

        # Ok, so this code runs every time you call the "compiled" function.
        if not self._input_format.is_type_for(input_data):
            raise TypeError("It looks like you have not been calling your function in a consistent manner.  Expected format: \n  {}, but got: \n  {}".format(self._input_format, NestedType.from_data(input_data)))
        arg_and_kwarg_values = self._input_format.get_leaves(input_data)
        # arg_and_kwarg_values = [a.get_value() if isinstance(a, SharedVariable) else a for a in arg_and_kwarg_values]  # Allows passing in Shared Variables

        shared_passed_in = [a for a in arg_and_kwarg_values if isinstance(a, SharedVariable)]
        assert shared_passed_in == self._shared_var_inputs, \
            "The shared variables you passed in, {}, Don't match the shared variables you passed in when you first called this compiled function: {}. " \
            "This creates problems for us.  Instead, compile your function a second time for the new shared inputs."\
            .format(['{}@{}'.format(repr(trace_value), hex(id(trace_value))) for trace_value in shared_passed_in], ['{}@{}'.format(repr(trace_value), hex(id(trace_value))) for trace_value in self._shared_var_inputs])
        arg_and_kwarg_values = [a for a in arg_and_kwarg_values if not isinstance(a, SharedVariable)]  # Remove shared variables from passed-in numeric args

        # Now, run the actual numeric function!
        if self._there_are_debug_variables:  # Need to take care of stripping off the debug variables
            all_out = self._compiled_fcn(*arg_and_kwarg_values)
            flat_output_data = all_out[:self._n_outputs]
            trace_out = all_out[self._n_outputs:self._n_outputs+self._n_trace_vars]
            local_out = all_out[self._n_outputs+self._n_trace_vars:]
            for trace_name, trace_value in izip_equal(self._trace_variable_keys, trace_out):
                CaptureTraceVariables.set_trace_value(trace_name, trace_value)
            trace_values = {k: v for k, v in zip(self._trace_variable_keys, trace_out)}
            # _TRACE_VALUES.update(trace_values)
            self._local_values = {k: v for k, v in zip(self._local_variable_keys, local_out)}
        else:  # Normal case
            flat_output_data = all_out = self._compiled_fcn(*arg_and_kwarg_values)
        true_out = self._output_format.expand_from_leaves(flat_output_data, check_types=False) if len(flat_output_data)>0 else ()

        if self._debug_print_shapes:
            if self._debug_print_shapes=='first':
                self._debug_print_shapes = False
            new_update_shapes = [old.get_value().shape for old, new in self._original_updates]
            PLATO_LOGGER.info("Shape info for running function {f}: \n  Inputs Shapes ({n_in}): {inp}\n  Output Shapes ({n_out}): {out}\n  Update Shapes ({n_up}): {updates}".format(
                f = self._original_fcn.fcn_str(),
                n_in = len(arg_and_kwarg_values),
                inp = str(', '.join([str(a.shape).replace(' ', '') if isinstance(a, np.ndarray) else () for a in arg_and_kwarg_values])),
                n_out = len(flat_output_data),
                out = str(true_out.shape).replace(' ', '') if isinstance(true_out, np.ndarray) else str(', '.join([str(a.shape).replace(' ', '') for a in all_out])),
                n_up = len(new_update_shapes),
                updates = str(', '.join([('%s->%s' % (os, ns)).replace(' ', '') for os, ns in zip(self._old_update_shapes, new_update_shapes)]))
            ))

        for c in self._callbacks:
            c()

        return true_out

    def reset(self):
        assert self.resettable, "If you want to reset the state of your compiled function, you must compile with f.compile(resettable=True)"
        if self.updated_variables is not None:  # If it is none, vars are already in their initial states.
            for shared_var, value in izip_equal(self.updated_variables, self._original_variable_values):
                shared_var.set_value(value)

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
    if isinstance(data, (list, tuple)) and all(isinstance(d, (np.ndarray, SharedVariable)) or np.isscalar(d) for d in data):
        return tuple(_data_to_tensor(d, name=None, cast_to_floatx=cast_to_floatx, add_test_value=add_test_value) for d in data)

    if isinstance(data, SharedVariable):
        # data = data.get_value()
        return data

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

    is_dtype = lambda x, dtype: (isinstance(x, (np.ndarray, csr_matrix)) and x.dtype == dtype) or (isinstance(dtype, type) and isinstance(x, dtype))
    is_float = lambda x: is_dtype(x, float) or is_dtype(x, 'float32') or is_dtype(x, 'float64')
    # Need to also downcast ints to int32 if floatX is float32, otherwise things like int_array.mean() return float64
    # objects, which (a) slows things down and (b) causes an error when you try to update 32-bit shared variabkles
    # with 64 bit values.
    dtype = \
        theano.config.floatX if (cast_to_floatx == 'all' or (cast_to_floatx=='float' and is_float(data))) else \
        'int32' if (cast_to_floatx=='float' and theano.config.floatX == 'float32' and is_dtype(data, int)) else \
        'int64' if isinstance(data, (bool, int)) else \
        'float64' if isinstance(data, float) else \
        'int8' if data.dtype==bool else \
        data.dtype
    if isinstance(data, csr_matrix):
        # Here we make a bunch of hacks to accomodate sparse matrices so that we don't have to change any of our other
        # code when handling them.   This was assembled in haste before a deadline.  Possibly it could be cleaner.  Probably.
        from theano import sparse
        tensor = sparse.csr_matrix(name='unnamed' if name is None else name, dtype=dtype, )
        if add_test_value is True:
            tensor.tag.test_value = data.astype(theano.config.floatX)
        elif add_test_value=='shape':
            tensor.ishape=data.shape
        # Do what theano couldn't and add the dot method to sparse
        def flattenit(var, ndim):
            assert var.indim == ndim, "This is a horrendous hack.  We don't actually flatten, we just check to see if it's the right shape.  It's not.  Also it needs test values on to work."
            return var
        sparse.SparseVariable.flatten = property(lambda self: lambda ndim: flattenit(self, ndim))
        sparse.SparseVariable.dot = property(lambda self: lambda other: theano.dot(self, other))

    else:
        tensor = TensorType(dtype, (None, )*ndim)(name)
        if add_test_value is True:
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


def printit(var_name, var_val):
    print '%s: %s' % (var_name, var_val)


name_counts = {}


def tdbprint(var, name = None):
    if name is None:
        # TODO: Get default by sneakily grabbing name from calling scope.
        name = '%s@%s' % (str(var), hex(id(var)))
    elif '%c' in name:
        name_counts[name] = 0 if name not in name_counts else name_counts[name] + 1
        num = 0 if name not in name_counts else name_counts[name]
        name = name.replace('%c', str(num))
    tdb_trace(var, name, callback = lambda: printit(var_name = name, var_val = CaptureTraceVariables.TRACE_VALUES[name]))


class CaptureTraceVariables(object):
    """
    Used to catch updates.  Usage:

    with StateCatcher() as sc:
        # Code here
    updates = sc.get_updates()  # A List<Tuple<SharedVariable, Variable>> contaning all updates in which add_update was called.
    """

    CURRENT_CATCHER = None
    TRACE_VALUES = OrderedDict()

    def __init__(self, swallow):
        """
        :param swallow: A boolean.
            True if you'd like to "swallow" all updates produced, which will prevent your updates from being applied,
              unless you get them (using StateCatcher.get_updates) and re-add them (using add_updates).
            False if you'd like to pass updates on to be applied when the function compiles.
        :return:
        """
        self.swallow = swallow
        self._trace_vars = OrderedDict()  # Dict <var:name, (variable, batch_in_scan)>

    def __len__(self):
        return len(self._trace_vars)

    def __enter__(self):
        self._outer_catcher = CaptureTraceVariables.CURRENT_CATCHER
        CaptureTraceVariables.CURRENT_CATCHER = self
        return self

    def __exit__(self, *args):
        CaptureTraceVariables.CURRENT_CATCHER = self._outer_catcher

    def get_callbacks(self):
        return [callback for var, batch_in_scan, callback in self._trace_vars.values() if callback is not None]

    def keys(self):
        return self._trace_vars.keys()

    def values(self):
        return [var for var, batch_in_scan, callback in self._trace_vars.values()]

    def add_trace(self, variable, name, batch_in_scan = False, callback = None):
        self._trace_vars[name] = (variable, batch_in_scan, callback)
        if self._outer_catcher is not None and not self.swallow:  # Allows for nested StateCatchers (outer ones do not have to worry about inner ones stealing their updates)
            self._outer_catcher.add_trace(variable=variable, name=name, batch_in_scan=batch_in_scan, callback=callback)

    def get_trace_variable_info(self):
        return self._trace_vars.copy()

    @classmethod
    def set_trace_value(cls, name, value):
        cls.TRACE_VALUES[name] = value

def get_tdb_traces():
    return CaptureTraceVariables.TRACE_VALUES


def get_trace_value(name):
    return CaptureTraceVariables.TRACE_VALUES[name]


def tdb_trace(var, name = None, callback = None, batch_in_scan = False):
    """
    Add a trace of a variable.  This will allow the variable to be accessable globally after the function has been called
    through the function get_tdb_traces()

    :param var: A symbolic variable
    :param name: The name that you like to use to refer to the variable.
    :param callback: Optionally, a callback to add at the end.
    :param batch_in_scan: If the trace is set in a scan loop, this variable decides whether to capture the full scan
        over values of the variable (which may be useful but also memory-consuming) or just the last one.
        False means just take the last value in the scan loop
        True means keep the whole batch of values that this variable takes on within the loop.
    """
    if name is None:
        # TODO: Get default by sneakily grabbing name from calling scope.
        name = '%s@%s' % (str(var), hex(id(var)))
    assert CaptureTraceVariables.CURRENT_CATCHER is not None, "You must be called from a symbolic function to add trace variables.  Make sure your function, or one calling it, is decorated with @symbolic"
    CaptureTraceVariables.CURRENT_CATCHER.add_trace(variable=var, name=name, batch_in_scan=batch_in_scan, callback=callback)


def clear_tdb_traces():
    CaptureTraceVariables.TRACE_VALUES.clear()


STATE_CATCHER = None


def _get_state_catcher():
    return STATE_CATCHER


def _set_state_catcher(val):
    global STATE_CATCHER
    STATE_CATCHER = val


def add_update(shared_var, new_val, accumulate = None):
    """
    Add a shared-variable update.  This will store an update, so that in your compiled function, your shared variable
    will be updated

    :param shared_var: A theano SharedVariable object
    :param new_val: The new value for this sharedvariable to take on (usually a TensorVariable)
    :param accumulate: If multiple updates are applied to the same variable, add them.
    """
    assert isinstance(shared_var, SharedVariable), 'shared_var must be a theano shared variable.'
    state_catcher = _get_state_catcher()
    assert state_catcher is not None, "You tried to add an update from a function that is not symbolic, and is not being called by a symbolic function."
    state_catcher.add_update(shared_var, new_val, accumulate=accumulate)


def add_updates(updates, accumulate = None):
    """
    Add multiple shared-variable updates.

    :param updates: Can be:
        A list of 2-tuples of (shared_var, new_value)
        A dict of shared_var -> new_value
    :param accumulate: If multiple updates are applied to the same variable, add them.
    """
    if isinstance(updates, dict):
        updates = updates.items()
    for shared_var, new_val in updates:
        add_update(shared_var, new_val, accumulate=accumulate)


def get_latest_update(shared_var):
    """
    Get the latest update to a shared variable, or just return the original variable otherwise.
    :param shared_var: A theano shared variable
    :return: The updated value, or just the shared var.
    """
    state_catcher = _get_state_catcher()
    assert state_catcher is not None, "This function must be called within a CaptureUpdates context"
    return state_catcher[shared_var] if shared_var in state_catcher else shared_var


class CaptureUpdates(object):
    """
    Used to catch updates.  Usage:

    with StateCatcher() as sc:
        # Code here
    updates = sc.get_updates()  # A List<Tuple<SharedVariable, Variable>> contaning all updates in which add_update was called.
    """

    def __init__(self, swallow = False):
        """
        :param swallow: A boolean.
            True if you'd like to "swallow" all updates produced, which will prevent your updates from being applied,
              unless you get them (using StateCatcher.get_updates) and re-add them (using add_updates).
            False if you'd like to pass updates on to be applied when the function compiles.
        :return:
        """
        self.swallow = swallow

    def __enter__(self):
        self._outer_catcher = _get_state_catcher()
        _set_state_catcher(self)
        self._updates = OrderedDict()
        return self

    def __exit__(self, *args):
        _set_state_catcher(self._outer_catcher)

    def __getitem__(self, shared_variable):
        return self._updates[shared_variable]

    def __contains__(self, item):
        return item in self._updates

    def add_update(self, shared_var, new_val, accumulate = None):

        if accumulate is None:
            accumulate = _ACCUMULATE_UPDATES

        if shared_var in self._updates:
            if accumulate:
                self._updates[shared_var] = self._updates[shared_var] + new_val - shared_var  # (w+dw1)+(w+dw2)-w = w+dw1+dw2
            else:
                raise AssertionError("You tried to update shared-variable %s with tensor %s, but you've already updated it with tensor %s.\nIf you want to accumulate both updates, call your update from inside a 'with AccumulateUpdates():'" % (shared_var, new_val, self._updates[shared_var]))
        else:
            self._updates[shared_var] = new_val
        if self._outer_catcher is not None and not self.swallow:  # Allows for nested StateCatchers (outer ones do not have to worry about inner ones stealing their updates)
            self._outer_catcher.add_update(shared_var, new_val)

    def get_updates(self, as_dict = False):
        return OrderedDict(self._updates.items()) if as_dict else self._updates.items()


StateCatcher = CaptureUpdates  # Backwards compatibility


_ACCUMULATE_UPDATES = False


class AccumulateUpdates():
    """
    Use this object to enable update accumulation... For example if some parameter w is being used to optimize two
    different objectives, you may want to add them: w_new = w + delta_w_1 + delta_w_2.  It's generally best to avoid
    this, and instead add the gradients, and update those with a single optimizer, but this we provide this anyway
    because we at Plato believe you should be able to hurt yourself if you want to.
    """

    def __enter__(self, ):
        global _ACCUMULATE_UPDATES
        self._oldstate = _ACCUMULATE_UPDATES
        _ACCUMULATE_UPDATES = True

    def __exit__(self, *args):
        global _ACCUMULATE_UPDATES
        _ACCUMULATE_UPDATES = self._oldstate


@contextmanager
def accumulate_updates():
    global _ACCUMULATE_UPDATES
    _oldstate = _ACCUMULATE_UPDATES
    _ACCUMULATE_UPDATES = True
    yield
    _ACCUMULATE_UPDATES = _oldstate



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


def initialize_param(initial_value, shape = None, name = None, cast_floats_to_floatX = True, **shared_kwargs):
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
        variable = theano.shared(typecast(initial_value), name = name, borrow = True, allow_downcast=True, **shared_kwargs)
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
        raise Exception("Don't know how to instantiate variable from data of format {}.  \nData: {}".format(NestedType.from_data(initial_value), initial_value, ))
    return variable, params, variable_shape


def create_shared_variable(initializer_fcn, shape = None, name = None, cast_floats_to_floatX = True, **shared_kwargs):
    """
    :param initializer_fcn: Can be:
        - An array.  It may be cast to floatX.  It's verified with shape if shape is provided
        - A function which takes the shape and turns it into the array.
        - A scalar, in which case it's broadcase over shape.
    :param shape: Either a tuple or an integer
    :return: A shared variable, containing the numpy array returned by the initializer.
    """
    shared_var, _, _ = initialize_param(initializer_fcn, shape = shape, name = name, cast_floats_to_floatX=cast_floats_to_floatX, **shared_kwargs)
    return shared_var


def create_shared_variable_from_zeros(shape, name = None, **shared_kwargs):
    """
    Create a share variable from an array of zeros.
    :param shape: The shape of the variable.
    :param name: (Optionally, the name)
    :param shared_kwargs: Other keyword args for shared variable construction
    :return: A theano shared variable.
    """
    assert name is None or isinstance(name, basestring)  # Mostly checks that you didn't accidentally call like create_shared_variable_from_zeros(3, 4)
    return create_shared_variable(initializer_fcn=np.zeros(shape), name=name, **shared_kwargs)


def create_constant(value, name=None, cast_floats_to_floatX=True):

    if isinstance(value, float) or value.dtype == 'float' and cast_floats_to_floatX:
        return tt.constant(value, name=name, dtype=theano.config.floatX)
    else:
        return tt.constant(value, name=name)


def initialize_constant(shape, fill_value, name=None, cast_floats_to_floatX=True):
    """
    Initialize a theano constant.  Cast floats to the dtype in theano.config.floatX
    :param fill_value: A scalar - the value to fill in
    :param shape: The shape (a tuple of real or symbolic integers)
    :param name:
    :return: A theano constant
    """

    if isinstance(fill_value, float) and cast_floats_to_floatX:
        return tt.zeros(shape, dtype=theano.config.floatX)+fill_value
    else:
        return tt.zeros(shape, dtype = type(fill_value))+fill_value
