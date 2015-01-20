from plato.interfaces.decorators import symbolic_stateless

__author__ = 'peter'


@symbolic_stateless
def multiply_by_two(x):
    return x*2


@symbolic_stateless
class MultiplyByTwo(object):

    def __call__(self, x):
        return x*2


class GenericClass(object):

    def __init__(self):
        self._factor = 2

    @symbolic_stateless
    def multiply_by_two(self, x):
        return x*self._factor


def test_decorator_schemes():

    f1 = multiply_by_two
    f2 = MultiplyByTwo()
    f3 = GenericClass().multiply_by_two

    assert f1.compiled(2) == 4
    assert f1.symbolic_standard.compiled(2) == (4, )
    assert f2.compiled(2) == 4
    assert f2.symbolic_standard.compiled(2) == (4, )
    assert f3.compiled(2) == 4
    assert f3.symbolic_standard.compiled(2) == (4, )

if __name__ == '__main__':

    test_decorator_schemes()
