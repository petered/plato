from artemis.experiments.experiment_record import experiment_function
from artemis.experiments.ui import browse_experiments
from artemis.general.test_mode import is_test_mode
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.predictors.train_and_test import train_and_test_online_predictor
from artemis.plotting.db_plotting import dbplot, hold_dbplots
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.optimizers import get_named_optimizer


__author__ = 'peter'


@experiment_function
def demo_mnist_mlp(
        minibatch_size = 10,
        learning_rate = 0.1,
        optimizer = 'sgd',
        hidden_sizes = [300],
        w_init = 'xavier',
        hidden_activations ='tanh',
        output_activation = 'linear',
        cost = 'softmax-xe',
        visualize_params = False,
        test_every = 1,
        n_epochs = 10,
        max_training_samples = None,
        max_test_samples = None,
        use_bias = True,
        onehot = False,
        rng = 1234,
        plot = False,
        ):
    """
    Train an MLP on MNIST and print the test scores as training progresses.
    """

    if is_test_mode():
        minibatch_size = 5
        n_epochs = 2.
        dataset = get_mnist_dataset(n_training_samples=30, n_test_samples=30)
    else:
        dataset = get_mnist_dataset(n_training_samples=max_training_samples, n_test_samples=max_test_samples)

    if onehot:
        dataset = dataset.to_onehot()

    optimizer = get_named_optimizer(name = optimizer, learning_rate=learning_rate)

    # Setup the training and test functions
    net = MultiLayerPerceptron.from_init(
            layer_sizes=[dataset.input_size]+hidden_sizes+[10],
            hidden_activations=hidden_activations,
            output_activation=output_activation,
            w_init = w_init,
            use_bias=use_bias,
            rng = rng,
            )
    predictor = GradientBasedPredictor(
        function = net,
        cost_function=cost,
        optimizer=optimizer
        )

    def vis_callback(info, score):
        with hold_dbplots():
            for l, layer in enumerate(net.layers):
                dbplot(layer.linear_transform.w.get_value().T.reshape(-1, 28, 28) if l==0 else layer.linear_transform.w.get_value(), 'w[{}]'.format(l))

    # Train and periodically report the test score.
    info_score_pair_sequence = train_and_test_online_predictor(
        dataset = dataset,
        train_fcn = predictor.train.compile(),
        predict_fcn = predictor.predict.compile(),
        minibatch_size=minibatch_size,
        n_epochs=n_epochs,
        test_epochs=('every', test_every),
        score_measure='percent_argmax_correct',
        test_callback=vis_callback if visualize_params else None
        )

    return info_score_pair_sequence


demo_mnist_mlp.add_variant('full-batch', minibatch_size = 'full', n_epochs = 1000)
demo_mnist_mlp.add_variant('deep', hidden_sizes=[500, 500, 500, 500])

# demo_mnist_mlp.get_variant('deep').run()
print demo_mnist_mlp.get_variant('deep').get_latest_record().get_log()


# X=demo_mnist_mlp.add_variant('mini-mnist', max_training_samples=1000, max_test_samples=1000, hidden_sizes=[100], n_epochs=100, visualize_params=True)
#
# X.add_variant('full-batch', minibatch_size = 'full', n_epochs = 1000)
#
# X.add_variant('L2-loss', cost='mse', onehot=True, learning_rate=0.01)
#
# demo_mnist_mlp.add_variant(hidden_sizes=[])


# if __name__ == '__main__':

    # browse_experiments()
