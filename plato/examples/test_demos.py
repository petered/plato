from plato.examples.demo_dbn import demo_dbn_mnist

__author__ = 'peter'


def test_demo_dbn_mnist():
    demo_dbn_mnist(plot = True, test_mode = True)


if __name__ == '__main__':
    test_demo_dbn_mnist()
