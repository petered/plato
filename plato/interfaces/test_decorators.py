from abc import abstractmethod
from pytest import raises
from plato.core import symbolic_simple, symbolic_updater, symbolic_standard, SymbolicFormatError, \
    tdb_trace, get_tdb_traces, symbolic, set_enable_omniscence, EnableOmbniscence, clear_tdb_traces
import pytest
import theano
import numpy as np

__author__ = 'peter'


def test_stateless_symbolic_function():

    # Case 1: Function
    @symbolic
    def multiply_by_two(x):
        return x*2

    f1 = multiply_by_two
    assert f1.compile()(2) == 4
    assert f1.to_format(symbolic_standard).compile()(2) == [4]

    # Case 2: Method
    class GenericClass(object):

        def __init__(self):
            self._factor = 2

        @symbolic
        def multiply_by_two(self, x):
            return x*self._factor

    obj = GenericClass()
    f2 = obj.multiply_by_two
    assert f2.compile()(2) == 4
    assert f2.to_format(symbolic_standard).compile()(2) == [4]

    # Case 3: Callable class
    @symbolic
    class MultiplyByTwo(object):

        def __init__(self):
            self._factor = 2

        def __call__(self, x):
            return x*self._factor

    f3 = MultiplyByTwo()
    assert f3.compile()(2) == 4
    assert f3.to_format(symbolic_standard).compile()(2) == [4]


def test_stateful_symbolic_function():

    @symbolic
    class Counter(object):

        def __init__(self, initial_value = 0):
            self._initial_value = initial_value

        def __call__(self):
            counter = theano.shared(np.zeros((), dtype = 'int')+self._initial_value)
            return counter, [(counter, counter+1)]

    c = Counter().compile()

    c1 = c()
    assert c1 == 0

    c2 = c()
    assert c2 == 1

    c()
    c3 = c()
    assert c3 == 3


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

    @symbolic_simple
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

    @symbolic_simple
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

    @symbolic_simple
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

    @symbolic_simple
    class Multiplier(object):

        def __init__(self, factor = 2):
            self._factor = float(factor)

        def __call__(self, x):
            return x*self._factor

        @symbolic_simple
        def inverse(self, y):
            return y/self._factor

    m = Multiplier(3)

    forward_fcn = m.compile()
    inverse_fcn = m.inverse.compile()

    out = forward_fcn(37)
    assert out == 37*3
    recon = inverse_fcn(out)
    assert recon == 37


# @pytest.mark.skipif(True, reason = "This was an old feature that's been superceded by variable traces.  We keep the test around in case we ever want to bring it back.")
def test_omniscence():
    """
    This test demonstrates a useful bit of evil that we've added to the
    framework.  It violates the basic idea of encapsulation, but is useful
    for debugging purposes.

    When you decorate a symbolic function and compile it in 'omniscent' mode,
    it records all internals of the function, and makes them available through
    the "locals" property.
    """

    with EnableOmbniscence():

        # Way 2
        @symbolic_simple
        def average(a, b):
            sum_a_b = a+b
            return sum_a_b/2.


        @symbolic_simple
        class Averager(object):

            def __call__(self, a, b):
                sum_a_b = a+b
                return sum_a_b/2.

        class TwoNumberOperator(object):

            @symbolic_simple
            def average(self, a, b):
                sum_a_b = a+b
                return sum_a_b/2.

        for k, op in [
                ('function', average),
                ('callable_class', Averager()),
                ('method', TwoNumberOperator().average),
                ('standard_function', average.to_format(symbolic_standard)),
                ('standard_callable_class', Averager().to_format(symbolic_standard)),
                ('standard_method', TwoNumberOperator().average.to_format(symbolic_standard))
                ]:

            if k != 'function':
                continue  # For now we've reduced the functionality of this hacky code.  We'll see if its useful before bringing it back.

            average_fcn = op.compile()

            mean = average_fcn(3, 6)
            assert mean == ([4.5] if k.startswith('standard_') else 4.5)
            assert average_fcn.locals()['sum_a_b'] == 9


def test_method_caching_bug():
    """
    Previously there was a bug in BaseSymbolicFunction.__get__ where dispatched
    methods were cached using the method-wrapper as a key rather than the instance.
    This caused the same method to be dispatched for different objects.  This test
    )catches that bug.  Before it was fixed, the second counter would appear to just
    continue the counting of the first, which is obviously not what you want.
    """
    class Counter(object):

        def __init__(self, initial_value = 0):
            self._count_var = theano.shared(np.array([initial_value]))

        @symbolic
        def count(self):
            return (self._count_var, ), [(self._count_var, self._count_var+1)]

        @symbolic
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

    @symbolic
    def average(a, b):
        sum_a_b = a+b
        tdb_trace(sum_a_b, name = 'sum_a_b')
        return sum_a_b/2.

    f = average.compile()

    assert f(3, 5) == 4
    assert get_tdb_traces()['sum_a_b'] == 8
    clear_tdb_traces()  # Needed due to unresolved thing where the drace callback happens on every symbolic function call in the future somehow.



def test_named_arguments():
    """
    We allow named arguments in Plato.  Note that you have to
    be consistent in your use of args and kwargs once you've compiled a funciton, otherwise
    you
    :return:
    """
    @symbolic
    def add_and_div(x, y, z):
        return (x+y)/z

    f = add_and_div.compile()
    assert f(2, 4, 3.) == 2
    with raises(TypeError):
        # You were inconsistent - used args the first time, kwargs the second.
        assert f(x=2, y=4, z=3.)
    f = add_and_div.compile()
    assert f(x=2, y=4, z=3.) == 2
    with raises(KeyError):
        # You were inconsistent - used args the first time, kwargs the second.
        assert f(2, 4, 3.)
    assert f(y=4, x=2, z=3.) == 2
    f = add_and_div.compile()
    assert f(2, y=4, z=3.) == 2


def test_strrep():
    """
    Just make sure that our wrappers communicate what types they're wrapping - otherwise it becomes a pain to debug.
    :return:
    """

    # Function
    @symbolic
    def do_thing(x):
        return x*3

    assert 'do_thing' in str(do_thing)
    f = do_thing.compile()
    assert 'do_thing' in str(f)

    # Callable class
    @symbolic
    class Thing(object):

        def __call__(self, x):
            return x*2

    t = Thing()
    assert 'Thing' in str(t)

    f_t = t.compile()
    assert 'Thing' in str(f_t)

    # Class with method
    class MultiplyBy(object):

        def __init__(self, x):
            self.x=x

        @symbolic
        def mult(self, y):
            return self.x*y

    m = MultiplyBy(3)
    assert 'MultiplyBy' in str(m.mult) and 'mult' in str(m.mult)
    f_m = m.mult.compile()
    assert 'MultiplyBy' in str(f_m) and 'mult' in str(f_m)


if __name__ == '__main__':
    test_strrep()
    test_omniscence()
    test_named_arguments()
    test_stateless_symbolic_function()
    test_stateful_symbolic_function()
    test_debug_trace()
    test_method_caching_bug()
    test_pure_updater()
    test_function_format_checking()
    test_callable_format_checking()
    test_inhereting_from_decorated()
    test_dual_decoration()
