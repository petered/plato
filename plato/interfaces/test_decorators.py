from abc import abstractmethod
from plato.interfaces.decorators import symbolic_stateless, symbolic_updater, symbolic_standard, SymbolicFormatError
import theano
import numpy as np

__author__ = 'peter'


def test_stateless_decorators():

    # Case 1: Function
    @symbolic_stateless
    def multiply_by_two(x):
        return x*2

    # Case 2: Method
    class GenericClass(object):

        def __init__(self):
            self._factor = 2

        @symbolic_stateless
        def multiply_by_two(self, x):
            return x*self._factor

    # Case 3: Callable class
    @symbolic_stateless
    class MultiplyByTwo(object):

        def __init__(self):
            self._factor = 2

        def __call__(self, x):
            return x*self._factor

    f1 = multiply_by_two
    assert f1.compile()(2) == 4
    assert f1.symbolic_standard.compile()(2) == [4]

    obj = GenericClass()
    f2 = obj.multiply_by_two
    assert f2.compile()(2) == 4
    assert f2.symbolic_standard.compile()(2) == [4]

    f3 = MultiplyByTwo()
    assert f3.compile()(2) == 4
    assert f3.symbolic_standard.compile()(2) == [4]


def test_standard_decorators():

    @symbolic_standard
    class Counter(object):

        def __init__(self, initial_value = 0):
            self._initial_value = initial_value

        def __call__(self):
            counter = theano.shared(np.zeros((), dtype = 'int')+self._initial_value)
            return (counter, ), [(counter, counter+1)]

    c = Counter().compile()

    c1 = c()
    assert c1 == [0]

    c2 = c()
    assert c2 == [1]

    c()
    c3 = c()
    assert c3 == [3]


def test_pure_updater():

    class MyThing(object):

        def __init__(self):
            self._var = theano.shared(0)

        def get_val(self):
            return self._var.get_value()

        @symbolic_updater
        def update(self):
            return [(self._var, self._var+1)]

    thing = MyThing()
    assert thing.get_val() == 0
    update_fcn = thing.update.compile()
    update_fcn()
    update_fcn()
    assert thing.get_val() == 2
    update_fcn = thing.update.compile()
    update_fcn()
    assert thing.get_val() == 3


def test_function_format_checking():

    @symbolic_stateless
    def good_format_thing(a, b):
        return a+b

    assert good_format_thing.compile()(3, 5) == 8

    @symbolic_standard
    def bad_format_thing(a, b):
        # This function has the standard decorator, but failes to return values in the standard format of (outputs, updates)
        return a+b

    try:
        bad_format_thing.compile()(3, 5)
        raise Exception('Failed to catch formatting Error')
    except SymbolicFormatError:
        pass


def test_callable_format_checking():

    @symbolic_stateless
    class GoodFormatThing(object):

        def __call__(self, a, b):
            return a+b

    assert GoodFormatThing().compile()(3, 5) == 8

    @symbolic_standard
    class BadFormatThing(object):

        def __call__(self, a, b):
            return a+b

    try:
        BadFormatThing().compile()(3, 5)
        raise Exception('Failed to catch formatting Error')
    except SymbolicFormatError:
        pass


def test_inhereting_from_decorated():

    @symbolic_stateless
    class AddSomething(object):

        def __call__(self, a):
            return a+self.amount_to_add()

        @abstractmethod
        def amount_to_add(self):
            pass

    class AddOne(AddSomething):

        def amount_to_add(self):
            return 1

    obj = AddOne()
    result = obj.compile()(2)
    assert result == 3


if __name__ == '__main__':

    test_stateless_decorators()
    test_standard_decorators()
    test_pure_updater()
    test_function_format_checking()
    test_callable_format_checking()
    # test_inhereting_from_decorated()
