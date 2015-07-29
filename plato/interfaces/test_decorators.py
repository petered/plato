from abc import abstractmethod
from plato.interfaces.decorators import symbolic_stateless, symbolic_updater, symbolic_standard, SymbolicFormatError, \
    tdb_trace, get_tdb_traces, set_enable_omniscence
import pytest
import theano
import numpy as np

__author__ = 'peter'


def test_stateless_decorators():

    # Case 1: Function
    @symbolic_stateless
    def multiply_by_two(x):
        return x*2

    f1 = multiply_by_two
    assert f1.compile()(2) == 4
    assert f1.symbolic_standard.compile()(2) == [4]

    # Case 2: Method
    class GenericClass(object):

        def __init__(self):
            self._factor = 2

        @symbolic_stateless
        def multiply_by_two(self, x):
            return x*self._factor

    obj = GenericClass()
    f2 = obj.multiply_by_two
    assert f2.compile()(2) == 4
    assert f2.symbolic_standard.compile()(2) == [4]

    # Case 3: Callable class
    @symbolic_stateless
    class MultiplyByTwo(object):

        def __init__(self):
            self._factor = 2

        def __call__(self, x):
            return x*self._factor

    f3 = MultiplyByTwo()
    assert f3.compile()(2) == 4
    assert f3.symbolic_standard.compile()(2) == [4]

    # Check that the types were correctly determined (igore this
    # if you're using this test as a tutorial - it's a detail)
    assert f1.get_decorated_type() == 'function'
    assert f1.symbolic_standard.get_decorated_type() == 'reformat'
    assert f2.get_decorated_type() == 'method'
    assert f2.symbolic_standard.get_decorated_type() == 'reformat'
    assert f3.get_decorated_type() == 'callable_class'
    assert f3.__call__.get_decorated_type() == 'method'
    assert f3.symbolic_standard.get_decorated_type() == 'reformat'


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
        """
        This function has the standard decorator, but fails to return values in the
        standard format of (outputs, updates)
        """
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


def test_omniscence():
    """
    This test demonstrates a useful bit of evil that we've added to the
    framework.  It violates the basic idea of encapsulation, but is useful
    for debugging purposes.

    When you decorate a symbolic function and compile it in 'omniscent' mode,
    it records all internals of the function, and makes them available through
    the "locals" property.
    """

    set_enable_omniscence(True)

    # Way 2
    @symbolic_stateless
    def average(a, b):
        sum_a_b = a+b
        return sum_a_b/2.


    @symbolic_stateless
    class Averager(object):

        def __call__(self, a, b):
            sum_a_b = a+b
            return sum_a_b/2.

    class TwoNumberOperator(object):

        @symbolic_stateless
        def average(self, a, b):
            sum_a_b = a+b
            return sum_a_b/2.

    for k, op in [
            ('function', average),
            ('callable_class', Averager()),
            ('method', TwoNumberOperator().average),
            ('standard_function', average.symbolic_standard),
            ('standard_callable_class', Averager().symbolic_standard),
            ('standard_method', TwoNumberOperator().average.symbolic_standard)
            ]:

        average_fcn = op.compile(mode = 'omniscent')
        average_fcn.set_debug_variables('locals')

        mean = average_fcn(3, 6)
        assert mean == ([4.5] if k.startswith('standard_') else 4.5)
        assert average_fcn.get_debug_values()['sum_a_b'] == 9


def test_method_caching_bug():
    """
    Previously there was a bug in BaseSymbolicFunction.__get__ where dispatched
    methods were cached using the method-wrapper as a key rather than the instance.
    This caused the same method to be dispatched for different objects.  This test
    catches that bug.  Before it was fixed, the second counter would appear to just
    continue the counting of the first, which is obviously not what you want.
    """
    class Counter(object):

        def __init__(self, initial_value = 0):
            self._count_var = theano.shared(np.array([initial_value]))

        @symbolic_standard
        def count(self):
            return (self._count_var, ), [(self._count_var, self._count_var+1)]

        @symbolic_stateless
        def get_count(self):
            return self._initial_value

    ca = Counter().count.compile()
    c1 = ca()
    assert c1 == [0]
    c2 = ca()
    assert c2 == [1]

    cb = Counter().count.compile()
    c1 = cb()
    assert c1 == [0]  # Before the fix, this was [2]


def test_debug_trace():
    """
    This demonstrates our debug system wherein we add traces
    to a global dict.
    :return:
    """

    @symbolic_stateless
    def average(a, b):
        sum_a_b = a+b
        tdb_trace(sum_a_b, name = 'sum_a_b')
        return sum_a_b/2.

    f = average.compile()

    assert f(3, 5) == 4
    assert f.get_debug_values()['sum_a_b'] == 8
    assert get_tdb_traces()['sum_a_b'] == 8


if __name__ == '__main__':
    test_debug_trace()
    test_method_caching_bug()
    test_omniscence()
    test_stateless_decorators()
    test_standard_decorators()
    test_pure_updater()
    test_function_format_checking()
    test_callable_format_checking()
    test_inhereting_from_decorated()
    test_dual_decoration()
