from functools import partial
import inspect
import logging
from abc import abstractproperty, abstractmethod
from general.local_capture import CaptureLocals, LocalsNotCapturedError
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

@symbolic_stateless: If the function just returns a single variable and does not update state.
@symbolic_updater: If the function returns only state updates.
@symbolic_standard: If the function returns (outputs, updates) as a tuple.

A decorated function has methods bound to it which allow it to be compiled and called in a standard format.
These methods are described in the ISymbolicFunction interface below.
"""

__author__ = 'peter'


ENABLE_OMNISCENCE = True


class ISymbolicFunction(object):

    def compile(self, **kwargs):
        """
        :return: A compiled version of function that takes and returns numpy arrays.

        Note: Compilation actually happens the first time the function is called, because it needs the inputs to tell it
        what kind of symbolic variables to instatiate.
        """

    @abstractproperty
    def symbolic_stateless(self):
        """
        :return: A function of the form:
            out = fcn(in_0, in_1, ...)
            Where out and in are tensors.  If the function cannot be cast to this form (for instance because it returns
            multiple outputs or updates) an exception will be raised when it is called.
        """

    @abstractproperty
    def symbolic_standard(self):
        """
        :return: A function of the form:
            (out_0, out_1, ...), ((shared_0, new_shared_0), (shared_1, new_shared_1), ...) = fcn(in_0, in_1, ...)
            Where all variables are symbolic, and shared_x variables are theano shared variables.
        """

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Call the function as it was defined, but do input/output type checking to confirm that it was implemented correctly
        """

    @abstractproperty
    def original(self):
        """
        Returns the original decorated function/method/class
        """

    @abstractmethod
    def locals(self):
        """
        Return a dictionary of variables INSIDE the symbolic funciton, at the time the return statement
        is executed.  This can be useful for debugging.
        """


class BaseSymbolicFunction(ISymbolicFunction):

    IS_DYNAMIC_CLASS = False

    def __new__(cls, *args, **kwargs):

        obj = object.__new__(cls)

        if issubclass(cls, BaseSymbolicFunction) and cls.IS_DYNAMIC_CLASS:
            # We're dealing with a callable class.  Since the decorated class can have its own
            # __init__ method, we need to sneakily call our own __init__ from here.

            BaseSymbolicFunction.__init__(obj, fcn = obj, instance = None)
        return obj

    def __init__(self, fcn, instance = None):

        self._fcn = fcn
        self._instance = instance
        self._locals = None
        self._dispatched_symbolic_methods = {}
        # Ok, there're basically 5 situations:
        # 1: This is an ordinary function
        #    inspect.isfunction: True
        #    instance is None
        # 2: This is a method before it has been bound to a class.
        #    inspect.isfunction: True
        #    instance is None
        #    The result of this seems to be discarded.
        # 3: This is a method
        #    inspect.isfunction: True
        #    instance is an object with class of the decorated method
        # 4: This is the __call__ function, which is decorated when its parent
        #    class is decorated.
        # 5: This is a decorated class
        #    inspect.isfunction: False
        #    instance is None
        #    IS_DYNAMIC_CLASS True
        # 6: This is a modified-format call of one of the above.
        #    inspect.isfunction: False
        #    instance: None

        # We need to distinguish between these situations, because locals need to be found
        # differently in each case.

        is_function = inspect.isfunction(fcn)
        is_method = inspect.ismethod(fcn)
        has_instance = instance is not None
        is_callable_class = hasattr(fcn, 'IS_DYNAMIC_CLASS') and fcn.IS_DYNAMIC_CLASS

        key = (is_function, is_method, has_instance, is_callable_class)

        if key == (False, False, False, False):
            raise Exception('I think you have to remove the decorator on some child of %s' % fcn._fcn.im_class.__name__)

        this_is_a = {
            (True, False, False, False): 'function',
            (False, True, False, False): 'reformat',
            (False, False, False, True): 'callable_class',
            (False, True, True, False): 'method',
            (True, False, True, False): 'method'
            }[key]
        self._type = this_is_a

    def _assert_is_tensor(self, arg, name):
        if not isinstance(arg, Variable):
            raise SymbolicFormatError('%s of function %s should have been a tensor, but was %s' % (name, self._fcn, arg))

    def _assert_all_tensors(self, args, name):
        if not (isinstance(args, (list, tuple)) and all(isinstance(arg, Variable) for arg in args)):
            raise SymbolicFormatError('%s of %s must a list/tuple of tensors.  They were %s instead' % (name, self._fcn, args, ))

    def _assert_all_updates(self, updates):
        if not (isinstance(updates, list) and all(len(up)==2 for up in updates) and
                all(isinstance(old, SharedVariable) and isinstance(new, Variable) for old, new in updates)):
            raise SymbolicFormatError('Updates from %s must be a list of 2-tuples of (shared_variable, update_tensor).  It was %s instead.  Did you forget to compile your function?' % (self._fcn, updates, ))

    def _assert_standard_return(self, return_val):
        if isinstance(return_val, SymbolicReturn):  # It's been checked already, you're clear.
            return
        else:  # Possibly remove this entirely and only allow SymbolicReturn
            if not (isinstance(return_val, tuple) and len(return_val)==2):
                raise SymbolicFormatError('Function %s was expected to return a 2-tuple of (outputs, updates) but returned %s instead' % (self._fcn, return_val))
            outputs, updates = return_val
            self._assert_all_tensors(outputs, 'Outputs')
            self._assert_all_updates(updates)

    def __get__(self, instance, owner):
        # What's going on here:
        # self is an ISymbolicFunction that wrapps a method - it is created at the time the class is, before
        # any object is instantiated.  Every time the method is requested from an instantiated object, this
        # function is called.  This function has 2 jobs: 1: Make sure the dispatched method is a symbolic function
        # of the same type as this (e.g. StatelessSymbolicFunction).  2: Make sure that each time the method is
        # requested for a particular instance, we return the same method.  2 is important for (a) efficiency - there's
        # no reason to create a separate object every time we want to get the method, and (b) debugging - because we
        # attach the local variables to the method, and want to get them later, so the returned method better have
        # the same address every time we request it.
        if instance in self._dispatched_symbolic_methods:
            # The caching is necessary for the .locals() method to work - we need to make
            # sure we're returning the same object every time a method is requested.
            dispatched_symbolic_method = self._dispatched_symbolic_methods[instance]
        else:
            dispatched_symbolic_method = self.__class__(self._fcn, instance=instance)
            self._dispatched_symbolic_methods[instance] = dispatched_symbolic_method
        return dispatched_symbolic_method

    def compile(self, **kwargs):
        return AutoCompilingFunction(self, **kwargs)

    def __call__(self, *args, **kwargs):
        """
        This function wraps a symbolic function.  Its job is to check that the format of the inputs/outputs are what we
        expect them to be, and also to capture the values of internal variables for inspection purposes, if we've
        enabled "omniscence".  If you're here because you're inspecting a stack-trace, you can probably just ignore this
        function.
        """

        self._check_inputs(*args, **kwargs)

        if self._instance is not None:  # Method calls - you need to pass the function
            args = (self._instance, )+args

        if ENABLE_OMNISCENCE:
            captor = CaptureLocals()
            with captor:
                return_val = self._fcn(*args, **kwargs)
            try:
                self._locals = captor.get_captured_locals()
            except LocalsNotCapturedError:
                logging.critical('Failed to capture locals for function %s.  This can happen when you call from debugger.' % (self._fcn, ))
        else:
            return_val = self._fcn(*args, **kwargs)

        self._check_outputs(return_val)
        return return_val

    @abstractmethod
    def _check_inputs(self, *args, **kwargs):
        pass

    @abstractmethod
    def _check_outputs(self, *args, **kwargs):
        pass

    def locals(self):
        if self._type in ('function', 'method'):
            local_vars = self._locals
        elif self._type == 'callable_class':
            local_vars = self.__call__._locals
        elif self._type == 'reformat':
            local_vars = self._fcn.__self__.locals()
        else:
            raise Exception('Unexpected type: %s' % (self._type))
        assert local_vars is not None, 'You tried to retrieve locals, but they are not available.  Have you called the function yet?'
        return LocalsContainer(local_vars)

    @abstractproperty
    def original(self):
        return self._fcn

    def get_decorated_type(self):
        return self._type


class LocalsContainer(object):
    """
    Just a dict that you can also reference by field.
    """

    def __init__(self, local_vars):
        for k, v in local_vars.iteritems():
            setattr(self, k, v)
        self._local_vars = local_vars

    def items(self):
        return self._local_vars.items()

    def iteritems(self):
        return self._local_vars.iteritems()

    def keys(self):
        return self._local_vars.keys()

    def values(self):
        return self._local_vars.values()

    def __getitem__(self, item):
        return self._local_vars[item]


class SymbolicStatelessFunction(BaseSymbolicFunction):
    """
    Use this to decorate a symbolic fcntion of the form:
    out = fcn(in_0, in_1, ...)    OR
    (out_0, out_1, ...) = fcn(in_0, in_1, ...)
    """

    def _check_inputs(self, *args, **kwargs):
        # TODO: Need to check inputs only if trying to compile as a theano function.
        # Previously we checked with the following lines that all inputs were symbolic:
        #   self._assert_all_tensors(args, 'Arguments')
        #   self._assert_all_tensors(kwargs.values(), 'Keyword Arguments')
        # ... But we've lifted that constraint - sometimes you want inputs defined at compile time.
        pass

    def _check_outputs(self, out):
        self._assert_is_tensor(out, 'Output')

    @property
    def symbolic_stateless(self):
        return self

    @property
    def symbolic_standard(self):
        return SymbolicStandardFunction(self._standard_function)

    def _standard_function(self, *args, **kwargs):
        out = self(*args, **kwargs)
        return (out, ), []


class SymbolicUpdateFunction(BaseSymbolicFunction):

    def _check_inputs(self, *args, **kwargs):
        pass

    def _check_outputs(self, out):
        self._assert_all_updates(out)

    @property
    def symbolic_stateless(self):
        raise Exception("Tried to get the symbolic_stateless function from an %s\n, which is a SymbolicUpdateFunction. - "
            "This won't work because updaters have state.")

    @property
    def symbolic_standard(self):
        return SymbolicStandardFunction(self._standard_function)

    def _standard_function(self, *args, **kwargs):
        updates = self(*args, **kwargs)
        return (), updates


class SymbolicStandardFunction(BaseSymbolicFunction):

    def _check_inputs(self, *args, **kwargs):
        pass

    def _check_outputs(self, out):
        self._assert_standard_return(out)

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


class SymbolicSingleOutSingleUpdate(BaseSymbolicFunction):

    def _check_inputs(self, *args, **kwargs):
        pass

    def _check_outputs(self, out):
        assert len(out) == 2, 'Output must consist of tensor, update'
        var, up = out
        self._assert_is_tensor(out)
        self._assert_all_updates([up])

    @property
    def symbolic_standard(self):
        raise NotImplementedError("Not done yet.  Need to refactor this whole thing anyway.")

    @property
    def symbolic_stateless(self):
        raise Exception("Cannot be caset to a stateless function")


class SymbolicSingleOutputUpdater(BaseSymbolicFunction):

    def _check_inputs(self, *args, **kwargs):
        pass

    def _check_outputs(self, out):
        assert len(out) == 2, 'Output must consist of tensor, update'
        var, up = out
        self._assert_is_tensor(out)
        self._assert_all_updates([up])

    @property
    def symbolic_standard(self):
        raise NotImplementedError("Not done yet.  Need to refactor this whole thing anyway.")

    @property
    def symbolic_stateless(self):
        raise Exception("Cannot be caset to a stateless function")


# TODO: Obviously the code in here is terrible.  Refactor it to So that there's only one
# SymbolicFunction class which takes format as an argument.  See branch redecoration
# TODO: Add option for "required fixed args": arguments that must be defined at compile time.
# Maybe like partial_symbolic_stateless(fixed_args = ['alpha', 'beta'])
# or symbolic_stateless.partial(fixed_args = ['alpha', 'beta'])

def symbolic_stateless(fcn):
    return _decorate_anything(SymbolicStatelessFunction, fcn)


def symbolic_standard(fcn):
    return _decorate_anything(SymbolicStandardFunction, fcn)


def symbolic_updater(fcn):
    return _decorate_anything(SymbolicUpdateFunction, fcn)


def symbolic_single_out_single_update(fcn):
    return _decorate_anything(SymbolicSingleOutSingleUpdate, fcn)


def symbolic_single_output_updater(fcn):
    return _decorate_anything(SymbolicSingleOutputUpdater, fcn)


def _decorate_anything(symbolic_function_class, callable_thing):
    """
    Decorate a callable thing as with a symbolic decorator

    # Cases to consider:
    # 1) Function: called directly with instance = None
    # 2) Method: Called from __get__ when the method is requested.  instance is the object to which the method is bound
    # 3) Callable class:
    """
    if inspect.isclass(callable_thing): # Case 3: Class with __call__ method
        return _decorate_callable_class(symbolic_function_class = symbolic_function_class, callable_class = callable_thing)
    else:  # Cases 1 and 2: Function or method
        return symbolic_function_class(callable_thing)


def _decorate_callable_class(symbolic_function_class, callable_class):

    assert hasattr(symbolic_function_class, '__call__'), "If you decorate a class with a symbolic decorator, it must "\
        "be callable.  If there's a specific method you want to decorate, decorate that instead."

    # print hasattr(callable_class.__call__, 'im_class')
    # if hasattr(callable_class.__call__, 'im_class'):
    #     assert callable_class.__call__.im_class is callable_class, 'If you decorate a class, the class itself must have ' \
    #         "a __call__ method.  If it's just the base-class, then just decorate that.  Bottom line: remove the decorator " \
    #         "from %s" % callable_class.__name__

    # Strategy 1: Return a new constructor that dynamically binds the function_type as a base-class when the object
    # is instantiated. (Now defunct - see git log if you want)

    # Strategy 2: Bind the function_type as a base-class to the class - the __new__ method of function_type will then be
    # called when the object is instantiated.
    class CallableSymbolicFunction(callable_class, symbolic_function_class):
        """
        This is a dynamic class that binds together the callable class with the symbolic function.  The idea is to make
        the callable class comply to the ISymbolicFunction interface.
        """

        IS_DYNAMIC_CLASS = True

        # Also decorate the __call__ method, so that type checking is done.
        __call__ = symbolic_function_class(callable_class.__call__)

        def __init__(self, *args, **kwargs):
            callable_class.__init__(self, *args, **kwargs)

        original = callable_class

    return CallableSymbolicFunction


class SymbolicFormatError(Exception):
    pass


class AutoCompilingFunction(object):
    """
    Given a Symbolic function, turn it into a compiled function that will accept and return numpy arrays.

    Actual compilation happens on the first use of the function, since it needs to see the arguments in order to
    instantiate the input tensors.
    """

    def __init__(self, fcn, cast_floats_to_floatX = True, mode = 'test_and_run', fixed_args = None, debug_getter = None):
        """
        :param fcn: A symbolic function (decorated with one of the above decorators)
        :param cast_floats_to_floatX: Case all floats to the global float type (define this in ~/.theanorc).
        :param mode: There are 3 modes:
            'run': Just compile and run - use this if you're confident in your code and just want to go.
            'test_and_run': Same as run, but you pass through test values once before compilation.  This lets you
                catch all sorts of errors.  You can also view test values by placing breakpoints, and viewing the
                value var.tag.test_value where var is some tensor variable.
            'debug': Never compile - just keep passing through test values.  This is basically like running the code
                in numpy, except to see variable values, you have to go var.tag.test_value
        :param fixed_args: A dict<arg_name: arg_value> of fixed arguments to the function.
        :return:
        """
        assert isinstance(fcn, ISymbolicFunction), 'You must pass a symbolic function.  Decorate it!'
        if mode == 'tr':
            mode = 'test_and_run'
        assert mode in ('run', 'test_and_run', 'debug', 'omniscent')

        self._fcn_class = type(fcn)
        self._fcn = fcn if fixed_args is None else partial(fcn, **fixed_args)
        self._format = format
        self._compiled_fcn = None
        self._cast_floats_to_floatX = cast_floats_to_floatX
        self._mode = mode
        self._debug_values = None
        self._debug_variable_getter = None
        self._debug_values = None

        self._callbacks = []
        if debug_getter is not None:
            self.set_debug_variables(debug_getter)
        if mode in ('test_and_run', 'debug', 'omniscent'):
            theano.config.compute_test_value = 'warn'
            __builtins__['showloc'] = show_all_locals
            __builtins__['locinfo'] = get_local_info

    def __call__(self, *args):
        """
        :param args, kwargs are the arguments that would go into fcn, but as real numpy arrays instead of symbols
        returns the result, in numpy arrays.
        """

        if self._compiled_fcn is None:

            d2t = partial(_data_to_tensor, cast_floats_to_floatx = self._cast_floats_to_floatX, test = self._mode in ('test_and_run', 'debug', 'omniscent'))

            tensor_args = [d2t(arg) for arg in args]

            return_value = self._fcn(*tensor_args)
            if issubclass(self._fcn_class, SymbolicStatelessFunction):
                outputs = return_value
                updates = []
            elif issubclass(self._fcn_class, SymbolicStandardFunction):
                outputs, updates = return_value
            elif issubclass(self._fcn_class, SymbolicUpdateFunction):
                outputs = ()
                updates = return_value
            elif issubclass(self._fcn_class, SymbolicSingleOutSingleUpdate):
                output, update = return_value
                outputs = (output, )
                updates = [update]
            elif issubclass(self._fcn_class, SymbolicSingleOutputUpdater):
                output, updates = return_value
                outputs = (output, )
            else:
                raise Exception("Get OUT!")

            # Now - check the trace variables.  If any of them are ancestors of the outputs,
            # add themj

            self._there_are_debug_variables = self._debug_variable_getter is not None or len(_TRACE_VARIABLES) > 0
            if self._there_are_debug_variables:

                all_debug_variables = {}
                self._single_output = not isinstance(outputs, (list, tuple))
                if self._single_output:
                    outputs = (outputs, )

                # Setup debug variables
                if self._debug_variable_getter is not None:
                    fetched_debug_variables = self._debug_variable_getter()
                    filtered_debug_variables = {k: v for k, v in fetched_debug_variables.iteritems() if isinstance(v, Variable)}
                    assert len(filtered_debug_variables) > 0, 'debug_variable_getter did not return any symbolic variables.' \
                        'It returned %s' % (filtered_debug_variables, )
                    all_debug_variables.update(filtered_debug_variables)

                # Add trace variables
                if len(_TRACE_VARIABLES) > 0:
                    # Find all trace variables that are ancestors of the output/updates
                    out_and_up = outputs + tuple(new for old, new in updates)
                    all_ancestors = set().union(*[find_all_ancestors(v) for v in out_and_up])

                    trace_variables = {name: var for name, var in _TRACE_VARIABLES.iteritems() if not find_all_ancestors(var).isdisjoint(all_ancestors)}
                    # Maybe check for name conflicts here?
                    all_debug_variables.update(trace_variables)
                    for name in trace_variables:
                        if name in _TRACE_CALLBACKS:
                            self._callbacks.append(_TRACE_CALLBACKS[name])

                # There may be some non-symbolic variables in the mix (like self).  We just filter these out.

                self._debug_variable_keys = all_debug_variables.keys()
                outputs_and_internals = tuple(outputs)+tuple(all_debug_variables.values())
                self._compiled_fcn = theano.function(inputs = tensor_args, outputs = outputs_and_internals, updates = updates, allow_input_downcast=self._cast_floats_to_floatX)
            elif self._mode == 'debug':  # Never compile - just keep passing through test values
                for (shared_var, new_val) in updates:  # Need to manually update shared vars
                    try:
                        shared_var.set_value(new_val.tag.test_value)
                    except AttributeError as err:
                        if err.message == "scratchpad instance has no attribute 'test_value'":
                            print 'Check - are you using randomstreams instead of shared_randomstreams?'
                        raise
                return [o.tag.test_value for o in outputs] if isinstance(outputs, (list, tuple)) else outputs.tag.test_value
            else:
                self._compiled_fcn = theano.function(inputs = tensor_args, outputs = outputs, updates = updates, allow_input_downcast=self._cast_floats_to_floatX)

        # Now, run the actual numeric function!
        if self._there_are_debug_variables:
            all_out = self._compiled_fcn(*args)
            self._debug_values = {k: v for k, v in zip(self._debug_variable_keys, all_out[-len(self._debug_variable_keys):])}
            _TRACE_VALUES.update(self._debug_values)
            numeric_output = all_out if len(self._debug_variable_keys) == 0 else all_out[:-len(self._debug_variable_keys)]
            if self._single_output:
                numeric_output, = numeric_output
        else:
            numeric_output = self._compiled_fcn(*args)

        for c in self._callbacks:
            c()

        return numeric_output

    def set_debug_variables(self, callback):
        """
        Define a callback that is called AFTER the graph is constructed.
        The callback should be of the form:

            dict<str: Variable> = callback()

        Where str is the name of each element, and Variables are symbolic variables
        linked to the graph.  After calling the function, you can retrieve the arrays
        associated with these variables through the method get_debug_values().

        You can also provide 'locals' as a callback.  In this case, the debug variables will be
        the locals of the symbolic function.
        """
        assert self._debug_variable_getter is None, 'You tried to set debug variables twice.  This remains ' \
            'banned until someone can provide a good reason for allowing it.'
        assert self._compiled_fcn is None, 'You can only set debug variables before the first call to this function.'
        if callback == 'locals':
            callback = self._fcn.locals
        elif callback == 'locals+class':
            callback = lambda: dict(self._fcn.locals().items() + [('self.'+k, v) for k, v in self._fcn.locals()['self'].__dict__.iteritems()])
        else:
            assert inspect.isfunction(callback), 'You can either provide a callback returning locals, or a string in ' \
                '{"locals", "class", "locals+class"}.  "%s" is not a valid argument.' % (callback, )

        self._debug_variable_getter = callback

    def get_debug_values(self):
        if self._debug_values is None:
            if self._debug_variable_getter is None:
                raise Exception('You need to define debug variables before requesting debug values.\n'
                    'See AutoCompilingFunction.set_debug_variables()')
            elif self._compiled_fcn is None:
                raise Exception('You need to run this function at lest once before requesting debug values')
            elif self._mode != 'omniscent':
                raise Exception("Function must be compiled in 'omniscent' mode to view debug variables.  It's in %s mode." % self._mode)
            else:
                raise Exception("I don't know why you're not getting an answer")
        return self._debug_values

    def add_callback(self, fcn):
        self._callbacks.append(fcn)

    def locals(self):
        return self._fcn.locals()

    @property
    def symbolic(self):
        """ Return the symbolic function """
        return self._fcn


def _is_symbol_or_value(var):
    return isinstance(var, ts.TensorType) or isinstance(var, np.ndarray) or np.isscalar(var)


def _data_to_tensor(data, name = None, cast_floats_to_floatx = True, test = True):
    # TODO:
    ndim = 0 if np.isscalar(data) else data.ndim

    is_dtype = lambda x, dtype: isinstance(x, dtype) or isinstance(x, np.ndarray) and x.dtype == dtype

    # Need to also downcast ints to int32 if floatX is float32, otherwise things like int_array.mean() return float64
    # objects, which (a) slows things down and (b) causes an error when you try to update 32-bit shared variabkles
    # with 64 bit values.

    dtype = \
        theano.config.floatX if (cast_floats_to_floatx and is_dtype(data, float)) else \
        'int32' if (cast_floats_to_floatx and theano.config.floatX == 'float32' and is_dtype(data, int)) else \
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


def find_all_ancestors(variable):
    """
    Return a set including the all ancestors of the given variable
    :param variable: A Theano Tensor
    :return: A set containing all ancestors, including the given variable.
    """

    return {variable}.union(*[find_all_ancestors(p) for p in variable.get_parents()])


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


_TRACE_VARIABLES = {}  # A dict of trace-variable-name: Trace Variable
_TRACE_VALUES = {}  # A dict of trace variable name: Most recently computed value
_TRACE_CALLBACKS = {}  # A dict of trace-variable-name: Callback to call after trace ver is used.


def get_tdb_traces():
    return _TRACE_VALUES


def tdb_trace(var, name = None, callback = None):
    if name is None:
        # TODO: Get default by sneakily grabbing name from calling scope.
        name = str(var)
    _TRACE_VARIABLES[name] = var
    if callback is not None:
        _TRACE_CALLBACKS[name] = callback


def set_enable_omniscence(state):
    """
    When ENABLE_OMNISCENCE is True, we do some weird things to the profiler to allow us to retrieve
    internal variables for debugging purposes.  Generally, this is harmless, but it seems that on occasion

    The error that happens in Theano is an AssertionError in vm.py
    assert c0 == sys.getrefcount(node_n_inputs)
    If you get this, just set_enable_omniscence(False).

    :param state: False to disable omniscence.
    """
    global ENABLE_OMNISCENCE
    ENABLE_OMNISCENCE = state
