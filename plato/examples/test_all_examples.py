from plato.examples.demo_mnist_mlp import demo_mnist_mlp
from plato.examples.demo_compare_optimizers import comparisons
from plato.tools.online_prediction.compare_symbolic_predictors import plot_records

__author__ = 'peter'


def test_demo_mnist_mlp():
    demo_mnist_mlp(test_mode = True)


def test_demo_compare_optimizers():

    comparison = comparisons.adamax_showdown(test_mode = True)
    records = comparison()
    plot_records(records, hang = False)


if __name__ == '__main__':

    test_demo_compare_optimizers()
    test_demo_mnist_mlp()
