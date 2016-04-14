from plato.tools.rbm.demo_rbm import demo_rbm_mnist
from plato.tools.rbm.demo_rbm_tutorial import demo_rbm_tutorial

__author__ = 'peter'


def test_demo_rbm_mnist():
    demo_rbm_mnist(plot = True)


def test_rbm_tutorial():
    demo_rbm_tutorial()