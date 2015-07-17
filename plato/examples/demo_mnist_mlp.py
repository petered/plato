import logging
from fileman.experiment_record import register_experiment, run_experiment
from general.test_mode import is_test_mode, set_test_mode
from plato.tools.cost import negative_log_likelihood_dangerous
from plato.tools.networks import MultiLayerPerceptron, normal_w_init
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent, get_named_optimizer
from plotting.db_plotting import dbplot
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import assess_online_predictor
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.mymath import sqrtspace

__author__ = 'peter'


def demo_mnist_mlp(
        minibatch_size = 10,
        learning_rate = 0.1,
        optimizer = 'sgd',
        hidden_sizes = [300],
        w_init_mag = 0.01,
        hidden_activation = 'tanh',
        output_activation = 'softmax',
        visualize_params = False,
        n_test_points = 30,
        n_epochs = 10,
        max_training_samples = None,
        ):
    """
    Train an MLP on MNIST and print the test scores as training progresses.
    """

    if is_test_mode():
        n_test_points = 3
        minibatch_size = 5
        n_epochs = 0.01
        dataset = get_mnist_dataset(n_training_samples=30, n_test_samples=30)
    else:
        dataset = get_mnist_dataset(n_training_samples=max_training_samples)

    if minibatch_size == 'full':
        minibatch_size = dataset.training_set.n_samples

    optimizer = get_named_optimizer(name = optimizer, learning_rate=learning_rate)

    # Setup the training and test functions
    predictor = GradientBasedPredictor(
        function = MultiLayerPerceptron(
            layer_sizes=hidden_sizes+[dataset.n_categories],
            input_size = dataset.input_size,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            w_init = normal_w_init(mag = w_init_mag)
            ),
        cost_function=negative_log_likelihood_dangerous,
        optimizer=optimizer
        ).compile()  # .compile() turns the GradientBasedPredictor, which works with symbolic variables, into a real one that takes and returns arrays.

    # vis_callback = lambda: dbplot(dict({
    #     'Layer[0].w': predictor.symbolic_predictor.layers[0].w.get_value().T.reshape(-1, 28, 28),
    #     'Layer[0].b': predictor.symbolic_predictor.layers[0].b.get_value(),
    #     }.items()+{}.items())

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
    name = 'MNIST-tanh-MLP[300,10]',
    function = lambda: demo_mnist_mlp(hidden_sizes=[300], hidden_activation='tanh', learning_rate=0.03),
    description='Baseline.  Gets 97.45% test score within 10 epochs.'
    )

register_experiment(
    name = 'MNIST1000_MLP[300,10]_minibatch-20',
    function = lambda: demo_mnist_mlp(learning_rate= 0.03, minibatch_size=20, max_training_samples=1000, n_epochs=1000)
    )

register_experiment(
    name = 'MNIST1000_MLP[300,10]_fullbatch',
    function = lambda: demo_mnist_mlp(learning_rate= 0.03, max_training_samples=1000, minibatch_size='full', n_epochs=10000)
    )

register_experiment(
    name = 'MNIST_MLP[300,10]_relu',
    function = lambda: demo_mnist_mlp(hidden_sizes=[300], hidden_activation= 'relu', optimizer = 'sgd', learning_rate=0.03),
    description='Try with rectified linear hidden units.',
    conclusion='Works nicely, gets 97.75% within 10 epochs.'
    )

register_experiment(
    name = 'MNIST_MLP[300,10]_norm-relu',
    function = lambda: demo_mnist_mlp(hidden_sizes=[300], hidden_activation= 'norm-relu', optimizer = 'sgd', learning_rate=2.),
    description='Try normalized-rectified linear units in an MLP',
    conclusion='Works ok-ish.  (gets around 95.5% in 10 epochs), just requires a much higher learning rate.'
    )



if __name__ == '__main__':

    which_experiment = 'MNIST_MLP[300,10]_norm-relu'
    set_test_mode(False)

    logging.getLogger().setLevel(logging.INFO)
    run_experiment(which_experiment)
