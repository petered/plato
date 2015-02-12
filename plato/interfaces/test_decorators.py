from abc import abstractmethod
import time
from plato.interfaces.decorators import symbolic_stateless, symbolic_updater, symbolic_standard, SymbolicFormatError
import pytest
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

    with pytest.raises(SymbolicFormatError):
        bad_format_thing.compile()(3, 5)


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

    with pytest.raises(SymbolicFormatError):
        BadFormatThing().compile()(3, 5)


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
    assert isinstance(obj, AddSomething)
    assert isinstance(obj, AddOne)


def test_dual_decoration():

    @symbolic_stateless
    class Multiplier(object):

        def __init__(self, factor = 2):
            self._factor = float(factor)

        def __call__(self, x):
            return x*self._factor

        @symbolic_stateless
        def inverse(self, y):
            return y/self._factor

    m = Multiplier(3)

    forward_fcn = m.compile()
    inverse_fcn = m.inverse.compile()

    out = forward_fcn(37)
    assert out == 37*3
    recon = inverse_fcn(out)
    assert recon == 37


def test_view_internals():
    """
    This test demonstrates a useful bit of evil that we've added to the
    framework.  It violates the basic idea of encapsulation, but is useful
    for debugging purposes.

    When you decorate a symbolic function and compile it in 'omniscent' mode,
    it records all internals of the function, and makes them available through
    the "locals" property.
    """

    t = time.time()

    @symbolic_stateless
    def average(a, b):
        sum_a_b = a+b
        return sum_a_b/2.

    average_fcn = average.compile(mode = 'omniscent')

    mean = average_fcn(3, 6)
    assert mean == 4.5
    assert average_fcn.locals['sum_a_b'] == 9

    print time.time() - t


if __name__ == '__main__':
    test_view_internals()
    test_stateless_decorators()
    test_standard_decorators()
    test_pure_updater()
    test_function_format_checking()
    test_callable_format_checking()
    test_inhereting_from_decorated()
    test_dual_decoration()
