from fileman.experiment_record import register_experiment, run_experiment
from general.test_mode import is_test_mode
from plato.tools.online_regressor import OnlineRegressor
from plato.tools.optimizers import get_named_optimizer
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import assess_online_predictor
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.mymath import sqrtspace

__author__ = 'peter'


def demo_mnist_mlp(
        minibatch_size = 10,
        learning_rate = 0.1,
        optimizer = 'sgd',
        regressor_type = 'multinomial',
        n_epochs = 20,
        n_test_points = 30,
        max_training_samples = None,
        include_biases = True,
        ):
    """
    Train an MLP on MNIST and print the test scores as training progresses.
    """

    if is_test_mode():
        n_test_points = 3
        minibatch_size = 5
        n_epochs = 0.01
        dataset = get_mnist_dataset(n_training_samples=30, n_test_samples=30, flat = True)
    else:
        dataset = get_mnist_dataset(n_training_samples=max_training_samples, flat = True)

    assert regressor_type in ('multinomial', 'logistic', 'linear')

    n_outputs = dataset.n_categories
    if regressor_type in ('logistic', 'linear'):
        dataset = dataset.to_onehot()

    predictor = OnlineRegressor(
        input_size = dataset.input_size,
        output_size = n_outputs,
        regressor_type = regressor_type,
        optimizer=get_named_optimizer(name = optimizer, learning_rate=learning_rate),
        include_biases = include_biases
        ).compile()

    # Train and periodically report the test score.
    results = assess_online_predictor(
        dataset=dataset,
        predictor=predictor,
        evaluation_function='percent_argmax_correct',
        test_epochs=sqrtspace(0, n_epochs, n_test_points),
        minibatch_size=minibatch_size
    )

    plot_learning_curves(results)


register_experiment(
    name = 'mnist-multinomial-regression',
    function = lambda: demo_mnist_mlp(regressor_type='multinomial'),
    description = 'Simple multinomial regression (a.k.a. One-layer neural network) on MNIST',
    conclusion = 'Gets to about 92.5'
    )

register_experiment(
    name = 'mnist-multinomial-regression-nobias',
    function = lambda: demo_mnist_mlp(regressor_type='multinomial', include_biases=False),
    description = 'Simple multinomial regression (a.k.a. One-layer neural network) on MNIST',
    conclusion = "Also gets to about 92.5.  So at least for MNIST you don't really need a bias term."
    )

register_experiment(
    name = 'mnist-linear-regression',
    function = lambda: demo_mnist_mlp(regressor_type='linear', learning_rate=0.01),
    description = 'Simple multinomial regression (a.k.a. One-layer neural network) on MNIST',
    conclusion = 'Requires a lower learning rate for stability, and then only makes it to around 86%'
    )

register_experiment(
    name = 'mnist-logistic-regression',
    function = lambda: demo_mnist_mlp(regressor_type='logistic'),
    description = 'Simple multinomial regression (a.k.a. One-layer neural network) on MNIST',
    conclusion = 'Gets just over 92%'
    )

if __name__ == '__main__':

    run_experiment('mnist-linear-regression')
