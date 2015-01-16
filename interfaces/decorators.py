from abc import abstractproperty

__author__ = 'peter'


class ISymbolicFunction(object):

    @abstractproperty
    def standard_symbolic_function(self):
        '''
        A function of the form:
        (out_0, out_1, ...), ((shared_0, new_shared_0), (shared_1, new_shared_1), ...) = fcn(in_0, in_1, ...)
        '''

    @abstractproperty
    def standard_compiled_function(self):
        '''
        A compiled version of the standard symbolic function
        '''

    @abstractproperty
    def compiled_function(self):
        '''
        The function in its compiled form.
        '''


class SymbolicStatelessFunction(ISymbolicFunction):
    '''
    Use this to decorate a symbolic function of the form:
    out = fcn(in_0, in_1, ...)    OR
    (out_0, out_1, ...) = fcn(in_0, in_1, ...)
    '''

    def __init__(self, fcn):
        self._fcn = fcn

    def __call__(self, *args, **kwargs):
        return self._fcn(*args, **kwargs)

    def compiled_function(self):
        return AutoCompilingFunction(self._fcn)




def stateless_symbolic_function(fcn):

    fcn.standard_compiled_fcn






class AutoCompilingFunction(object):

    def __init__(self, fcn, format = 'auto'):
        self._fcn = fcn
        self._format = format
        self._compiled_fcn = None

    def __call__(self, *args, **kwargs):
        '''
        :param args, kwargs are the arguments that would go into fcn, but as real numpy arrays instead of symbols
        returns the result, in numpy arrays.
        '''
        if self._compiled_fcn is None:



