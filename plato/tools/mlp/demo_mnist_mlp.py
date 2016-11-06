import logging

import theano.tensor as tt
import numpy as np

from artemis.fileman.experiment_record import register_experiment, run_experiment
from artemis.general.test_mode import is_test_mode, set_test_mode
from plato.tools.mlp.mlp import MultiLayerPerceptron
from artemis.plotting.db_plotting import dbplot
from artemis.ml.predictors import plot_learning_curves
from utils.benchmarks.predictor_comparison import assess_online_predictor
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.optimization.optimizers import get_named_optimizer
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.general.mymath import sqrtspace


__author__ = 'peter'


def demo_mnist_mlp(
        minibatch_size = 10,
        learning_rate = 0.1,
        optimizer = 'sgd',
        hidden_sizes = [300],
        w_init = 0.01,
        hidden_activation = 'tanh',
        output_activation = 'softmax',
        cost = 'nll-d',
        visualize_params = False,
        n_test_points = 30,
        n_epochs = 10,
        max_training_samples = None,
        use_bias = True,
        onehot = False,
        rng = 1234,
        plot = False,
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

    if onehot:
        dataset = dataset.to_onehot()

    if minibatch_size == 'full':
        minibatch_size = dataset.training_set.n_samples

    optimizer = get_named_optimizer(name = optimizer, learning_rate=learning_rate)

    # Setup the training and test functions
    predictor = GradientBasedPredictor(
        function = MultiLayerPerceptron.from_init(
            layer_sizes=[dataset.input_size]+hidden_sizes+[10],
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            w_init = w_init,
            use_bias=use_bias,
            rng = rng,
            ),
        cost_function=cost,
        optimizer=optimizer
        ).compile()  # .compile() turns the GradientBasedPredictor, which works with symbolic variables, into a real one that takes and returns arrays.

    def vis_callback(xx):
        p = predictor.symbolic_predictor._function
        in_layer = {
            'Layer[0].w': p.layers[0].linear_transform._w.get_value().T.reshape(-1, 28, 28),
            'Layer[0].b': p.layers[0].linear_transform._b.get_value(),
            }
        other_layers = [{'Layer[%s].w' % (i+1): l.linear_transform._w.get_value(), 'Layer[%s].b' % (i+1): l.linear_transform._b.get_value()} for i, l in enumerate(p.layers[1:])]
        dbplot(dict(in_layer.items() + sum([o.items() for o in other_layers], [])))

    # Train and periodically report the test score.
    results = assess_online_predictor(
        dataset=dataset,
        predictor=predictor,
        evaluation_function='percent_argmax_correct',
        test_epochs=sqrtspace(0, n_epochs, n_test_points),
        minibatch_size=minibatch_size,
        test_callback=vis_callback if visualize_params else None
    )

    if plot:
        plot_learning_curves(results)


register_experiment(
    name = 'MNIST-tanh-MLP[300,10]',
    function = lambda: demo_mnist_mlp(hidden_sizes=[300], hidden_activation='tanh', learning_rate=0.03),
    description='Baseline.  Gets 97.45% test score within 10 epochs.'
    )

register_experiment(
    name = 'MNIST1000_onelayer_minibatch-20',
    function = lambda: demo_mnist_mlp(learning_rate= 0.03, hidden_sizes = [], minibatch_size=20, max_training_samples=1000, n_epochs=1000),
    description='How does a single-layer (logistic-regression) net do on MNIST-1000?',
    conclusion='Gets to about 87.5% before overfitting its way down to 86.7.  100% on training.'
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
    name = 'MNIST_MLP[300,10]_all_relu',
    function = lambda: demo_mnist_mlp(hidden_sizes=[300], hidden_activation= 'relu', output_activation='relu',
        optimizer = 'sgd', learning_rate=0.03, onehot = True, cost = 'mse'),
    description='Try with rectified linear hidden units as the ONLY unit type.',
    conclusion='Works very nicely, gets 98.27% within 10 epochs.'
    )

register_experiment(
    name = 'MNIST_MLP[300,10]_all_relu-nobias',
    function = lambda: demo_mnist_mlp(hidden_sizes=[300], hidden_activation= 'relu', output_activation='relu',
        optimizer = 'sgd', learning_rate=0.03, onehot = True, cost = 'mse', use_bias = False),
    description='Well since all-relu network works so nicely, lets see how it holds up with no biases.',
    conclusion="We don't need no biases.  98.26%, so we lose 0.01% by not having them.  This gives us the nive property "
        "of total scale invariance."
    )

register_experiment(
    name = 'MNIST_MLP[300,10]_norm-relu',
    function = lambda: demo_mnist_mlp(hidden_sizes=[300], hidden_activation= 'norm-relu', optimizer = 'sgd', learning_rate=2.),
    description='Try normalized-rectified linear units in an MLP',
    conclusion='Works ok-ish.  (gets around 95.5% in 10 epochs), just requires a much higher learning rate.'
    )

register_experiment(
    name = 'MNIST_multiplicitive_sgd',
    function = lambda: demo_mnist_mlp(
        hidden_sizes=[],
        hidden_activation='tanh',
        w_init=lambda *shape: 0.1*np.ones(shape),
        output_activation='softmax',
        optimizer = 'mulsgd',
        learning_rate=0.1,
        visualize_params=True
        ),
    description='Does multiplicitive SGD work for a one layer network?',
    conclusion='Yes!'
    )

register_experiment(
    name = 'MNIST_MLP[300,10]_multiplicitive_sgd',
    function = lambda: demo_mnist_mlp(
        hidden_sizes=[300],
        hidden_activation='tanh',
        w_init=lambda *shape: 0.01*(np.random.randn(*shape)**2),
        output_activation='softmax',
        optimizer = 'mulsgd',
        learning_rate=0.1,
        visualize_params=True,
        n_epochs=20
        ),
    description='Does multiplicitive SGD work for a multi-layer network?',
    conclusion='Well kind of.  It seems to learn ridiculously sparse weights.  Score gets up to 95% in 20 epochs'
    )


register_experiment(
    name = 'MNIST-tanh-MLP[300,10]-cos-cost',
    function = lambda: demo_mnist_mlp(hidden_sizes=[300], hidden_activation='tanh', output_activation='softmax',
        learning_rate=0.03, cost='cos', onehot=True),
    description='What happens when we use cosine-distance as a cost?',
    conclusion='Seems to work - 95.7%.  May benefit some learning-rate-fiddling to get it better.'
    )


register_experiment(
    name = 'MNIST-relu-explode',
    function = lambda: demo_mnist_mlp(hidden_sizes=[100], hidden_activation= 'relu', output_activation='relu',
        optimizer = 'sgd', learning_rate=0.1, onehot = True, cost = 'mse', use_bias = False, w_init=0.01, n_epochs=10),
    description='Here we try to find the parameters that will reveal the RELU exploding problem.',
    conclusion=""
    )

register_experiment(
    name = 'mnist_mlp_leaky_relu',
    description='So instead of a ReLU we can use a "bent" unit.  These guys ',
    function = lambda alpha, **kwargs: demo_mnist_mlp(hidden_activation = lambda x: tt.switch(x > 0, x, x*alpha),
        optimizer = 'sgd', learning_rate=0.03, **kwargs),
    versions = dict(
        bent = dict(alpha = 0.5),
        abs = dict(alpha = -1),
        relu = dict(alpha = 0),
        small = dict(alpha = 0.05),
        deep_relu = dict(alpha = 0.05, hidden_sizes = [300, 300]),
        deep_abs = dict(alpha = -1, hidden_sizes = [300, 300]),

    ),
    current_version = 'deep_abs',
    conclusion="""
        relu: 97.78
        bent: 96.04
        abs: 98.16
        small: 97.67

        deep_relu: 97.76
        deep_abs: 97.91

    """
    )


if __name__ == '__main__':

    which_experiment = 'mnist_mlp_leaky_relu'
    set_test_mode(False)

    logging.getLogger().setLevel(logging.INFO)
    run_experiment(which_experiment)
