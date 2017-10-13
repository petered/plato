from abc import abstractmethod

from artemis.general.hashing import compute_fixed_hash, fixed_hash_eq
from plato.interfaces.helpers import create_shared_variable, shared_like
from plato.tools.common.config import hold_float_precision
from pytest import raises
from plato.core import symbolic_simple, symbolic_updater, SymbolicFormatError, \
    tdb_trace, get_tdb_traces, symbolic, set_enable_omniscence, EnableOmniscence, clear_tdb_traces, add_update, \
    symbolic_multi, symbolic_stateless, create_shared_variable
import pytest
import theano
import theano.tensor as tt
import numpy as np

__author__ = 'peter'


def test_stateless_symbolic_function():

    # Case 1: Function
    @symbolic
    def multiply_by_two(x):
        return x*2

    f1 = multiply_by_two
    assert f1.compile()(2) == 4
    assert f1.to_format(symbolic_multi).compile()(2) == (4, )

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
    assert f2.to_format(symbolic_multi).compile()(2) == (4, )

    # Case 3: Callable class
    @symbolic
    class MultiplyByTwo(object):

        def __init__(self):
            self._factor = 2

        def __call__(self, x):
            return x*self._factor

    f3 = MultiplyByTwo()
    assert f3.compile()(2) == 4
    assert f3.to_format(symbolic_multi).compile()(2) == (4, )


def test_stateful_symbolic_function():

    @symbolic
    class Counter(object):

        def __init__(self, initial_value = 0):
            self._initial_value = initial_value

        def __call__(self):
            counter = theano.shared(np.zeros((), dtype = 'int')+self._initial_value)
            add_update(counter, counter+1)
            return counter

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

        @symbolic
        def update(self):
            add_update(self._var, self._var+1)
            # return [(self._var, self._var+1)]

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

    @symbolic_multi
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

    @symbolic_multi
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

    with EnableOmniscence():

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
                ('standard_function', average.to_format(symbolic_multi)),
                ('standard_callable_class', Averager().to_format(symbolic_multi)),
                ('standard_method', TwoNumberOperator().average.to_format(symbolic_multi))
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
            add_update(self._count_var, self._count_var+1)
            return self._count_var

        @symbolic
        def get_count(self):
            return self._initial_value

    ca = Counter().count.compile()
    c1 = ca()
    assert c1 == 0
    c2 = ca()
    assert c2 == 1

    cb = Counter().count.compile()
    c1 = cb()
    assert c1 == 0  # Before the fix, this was [2]


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
    with raises(TypeError):
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


def test_scan():

    @symbolic
    def running_sum(x):
        s = create_shared_variable(0.)
        new_s = s+x
        add_update(s, new_s)
        return new_s

    @symbolic
    def cumsum_and_remember(arr):
        return running_sum.scan(sequences = [arr])

    f = cumsum_and_remember.compile()

    ar = np.random.randn(10)
    csum = f(ar)
    assert np.allclose(csum, np.cumsum(ar), atol=1e-6)
    more_csum = f(ar)
    assert np.allclose(more_csum, csum[-1]+np.cumsum(ar), atol=1e-6)


def test_catch_non_updates():

    var = create_shared_variable(0)

    @symbolic_updater
    def lying_function_that_says_its_an_updater_but_isnt():
        pass

    f = lying_function_that_says_its_an_updater_but_isnt.compile()
    with raises(SymbolicFormatError):
        f()

    @symbolic_updater
    def honest_function_that_actually_updates():
        add_update(var, var+1)

    g = honest_function_that_actually_updates.compile()
    g()
    g()
    assert var.get_value() == 2


def test_catch_sneaky_updates():

    var = create_shared_variable(0)

    @symbolic_stateless
    def lying_function_that_says_its_stateless_but_has_state():
        add_update(var, var+1)
        return var+1

    f = lying_function_that_says_its_stateless_but_has_state.compile()

    with raises(SymbolicFormatError):
        f()

    assert var.get_value() == 0

    @symbolic_stateless
    def honest_function_that_actually_is_stateless():
        return var+1

    g = honest_function_that_actually_is_stateless.compile()
    assert g() == 1
    assert g() == 1


def test_ival_ishape():
    """
    ival, ishape, idim, idtype give the initial values, shapes, number of dimensions, and data types
    of theano variables.  These properties are useful for input checking and debugging.
    """

    @symbolic
    def mat_mult(a, b):
        assert a.indim == 2 and b.indim == 2
        assert a.idtype == theano.config.floatX and b.idtype == theano.config.floatX, 'We only take floats around these parts.'
        assert a.ishape[1] == b.ishape[0], 'Matrices not aligned!'
        c = a.dot(b)
        assert c.ishape == (a.ishape[0], b.ishape[1])
        return c

    rng = np.random.RandomState(1234)

    foo = rng.randn(3, 4)
    bar = rng.randn(3, 3)
    baz = rng.randn(4, 5)
    hap = rng.randint(255, size = (4, 5))

    f = mat_mult.compile(add_test_values = True)
    with raises(AssertionError):
        z = f(foo, bar)

    f = mat_mult.compile(add_test_values = True)
    with raises(AssertionError):
        z = f(foo, hap)

    z = f(foo, baz)
    assert np.allclose(z, foo.dot(baz))


def test_named_outputs():
    @symbolic
    def do_some_ops(x):
        return {'cos': tt.cos(x), 'sin': tt.sin(x), 'exp': tt.exp(x), 'log': tt.log(x)}

    f = do_some_ops.compile()
    rng = np.random.RandomState(0)
    x = rng.rand(10)*np.pi*2
    out = f(x)
    assert np.allclose(out['cos'], np.cos(x))
    assert np.allclose(out['sin'], np.sin(x))
    assert np.allclose(out['exp'], np.exp(x))
    assert np.allclose(out['log'], np.log(x))


def test_named_outputs_with_trace():
    """
    Follows a different code path than just named outputs, so we test again.
    """

    @symbolic
    def do_some_ops(x):
        tdb_trace(tt.tan(x), 'tan(x)')
        return {'cos': tt.cos(x), 'sin': tt.sin(x), 'exp': tt.exp(x), 'log': tt.log(x)}

    f = do_some_ops.compile(add_test_values = True)
    rng = np.random.RandomState(0)
    x = rng.rand(10)*np.pi*2
    out = f(x)
    assert np.allclose(out['cos'], np.cos(x))
    assert np.allclose(out['sin'], np.sin(x))
    assert np.allclose(out['exp'], np.exp(x))
    assert np.allclose(out['log'], np.log(x))
    assert np.allclose(get_tdb_traces()['tan(x)'], np.tan(x))


def test_arbitrary_structures():

    with hold_float_precision(64):
        @symbolic
        def my_func(inp):
            """
            :param a: A list of 2-tuples
            :return: A dict of keys: lists
            """
            return {'a': [inp[0][0], inp[1][0]], 'b':[inp[0][1], inp[1][1]]}

        f = my_func.compile()

        rng = np.random.RandomState(1234)
        inputs = [(rng.randn(2, 3), rng.randn(2, 3)), (rng.randn(2, 3), rng.randn(2, 3))]
        out = f(inputs)

        assert fixed_hash_eq(out, {'a': [inputs[0][0], inputs[1][0]], 'b': [inputs[0][1], inputs[1][1]]})


def test_shared_input():
    """
    Needed because theano doesn't by default allow passing shared variables in to a function.  Internally we have a
    work-around that we test here.
    """

    @symbolic
    def increment(a_):
        new_val = a_ + 1
        add_update(a_, new_val)
        return new_val

    f_inc = increment.compile()
    a = create_shared_variable(2)
    f_inc(a)
    assert a.get_value()==3
    f_inc(a)
    assert a.get_value()==4
    b = create_shared_variable(6)

    with raises(AssertionError):
        f_inc(b)
    # assert b.get_value()==7  # This line would have failed had we not had the assertion error

    assert b.get_value()==6
    f_incb = increment.compile()
    f_incb(b)
    assert b.get_value()==7


def test_function_reset():

    @symbolic
    def running_sum(x):
        s = create_shared_variable(0)
        new_s = s+x
        add_update(s, new_s)
        return new_s

    f = running_sum.compile(resettable=True)

    assert np.array_equal([f(x) for x in [1, 2, 3]], [1, 3, 6])
    assert np.array_equal([f(x) for x in [1, 2, 3]], [7, 9, 12])
    f.reset()
    assert np.array_equal([f(x) for x in [1, 2, 3]], [1, 3, 6])


def test_trace_var_in_scan():

    @symbolic
    def running_sum(x):
        s = create_shared_variable(0)
        new_s = s+x
        add_update(s, new_s)
        tdb_trace(x**2, 'x_in_loop')
        tdb_trace(x**3, 'x_in_loop_catch_all', batch_in_scan=True)
        return new_s

    @symbolic
    def my_cumsum(x):

        tdb_trace(x**2, name='x_out_of_loop')
        return running_sum.scan(
            sequences = [x]
            )

    f = my_cumsum.compile()
    assert np.array_equal(f(np.arange(4)), [0, 1, 3, 6])
    assert np.array_equal(get_tdb_traces()['x_out_of_loop'], np.arange(4)**2)
    assert np.array_equal(get_tdb_traces()['x_in_loop'], 3**2)
    assert np.array_equal(get_tdb_traces()['x_in_loop_catch_all'], np.arange(4)**3)


def test_easy_scan_syntax():

    @symbolic
    def accumulator(v, shape):
        accum = create_shared_variable(np.zeros(shape))
        new_accum = accum + v
        add_update(accum, new_accum)
        return new_accum

    x = np.random.randn(5, 3)
    f = accumulator.partial(shape=x.shape[1:]).scan.compile()

    assert np.allclose(f(x), np.cumsum(x, axis=0))


def test_scan_no_return():

    state = create_shared_variable(np.zeros(()))

    @symbolic
    def do_something_internal(a, b):
        new_state = state+ a*b
        add_update(state, new_state)

    out = do_something_internal.scan.compile()(np.arange(6).astype(float), np.arange(1,7).astype(float))

    assert out is None
    assert np.array_equal(state.get_value(), np.arange(6).dot(np.arange(1, 7)))



if __name__ == '__main__':
    test_ival_ishape()
    test_catch_sneaky_updates()
    test_catch_non_updates()
    test_scan()
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
    test_named_outputs()
    test_named_outputs_with_trace()
    test_arbitrary_structures()
    test_shared_input()
    test_function_reset()
    test_trace_var_in_scan()
    test_easy_scan_syntax()
    test_scan_no_return()