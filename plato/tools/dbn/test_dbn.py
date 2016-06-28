from artemis.general.test_mode import set_test_mode
from plato.tools.dbn.demo_dbn import demo_dbn_mnist

__author__ = 'peter'


def test_demo_dbn_mnist():
    demo_dbn_mnist(plot = True)


if __name__ == "__main__":
    set_test_mode(True)
    test_demo_dbn_mnist()
