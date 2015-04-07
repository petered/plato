from plato.examples.demo_prediction_example import compare_example_predictors
from plato.examples.demo_mnist_mlp import demo_mnist_mlp
from plato.examples.demo_compare_optimizers import comparisons
from plato.tools.online_prediction.compare_symbolic_predictors import plot_records
from plato.examples.demo_dbn import demo_dbn_mnist
from plato.examples.demo_rbm import demo_rbm_mnist
__author__ = 'peter'


def test_demo_mnist_mlp():
    demo_mnist_mlp(test_mode = True)


def test_demo_compare_optimizers():

    comparison = comparisons.adamax_showdown(test_mode = True)
    records = comparison()
    plot_records(records, hang = False)


def test_demo_dbn_mnist():
    demo_dbn_mnist(plot = True, test_mode = True)


def test_demo_rbm_mnist():
    demo_rbm_mnist(plot = True, test_mode = True)


def test_demo_compare_optimizers():
    from plato.examples.demo_compare_optimizers import comparisons

    for name, c in comparisons.__dict__.iteritems():
        _ = c(test_mode = True)


def test_demo_prediction_example():

    compare_example_predictors(test_mode = True)


if __name__ == '__main__':

    test_demo_prediction_example()
    test_demo_compare_optimizers()
    test_demo_mnist_mlp()
    test_demo_rbm_mnist()
    test_demo_dbn_mnist()
