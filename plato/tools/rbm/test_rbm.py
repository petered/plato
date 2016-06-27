from artemis.general.test_mode import set_test_mode
from plato.tools.rbm.demo_rbm import demo_rbm_mnist
from plato.tools.rbm.demo_rbm_tutorial import demo_rbm_tutorial

__author__ = 'peter'


def test_demo_rbm_mnist():
    demo_rbm_mnist(plot = True)


def test_rbm_tutorial():
    demo_rbm_tutorial()


if __name__ == '__main__':
    set_test_mode(True)
    test_demo_rbm_mnist()
    test_rbm_tutorial()
