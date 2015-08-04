import logging

from general.test_mode import is_test_mode
from plato.tools.optimization.cost import negative_log_likelihood_dangerous
from plato.tools.mlp.networks import MultiLayerPerceptron, normal_w_init
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent
from utils.benchmarks.train_and_test import percent_argmax_correct
from utils.datasets.mnist import get_mnist_dataset


__author__ = 'peter'


def demo_mnist_mlp(
        test_period = 1000,
        minibatch_size = 10,
        eta = 0.1,
        hidden_sizes = [300],
        w_init_mag = 0.01,
        hidden_activation = 'sig',
        n_epochs = 10,
        max_training_samples = None,
        ):
    """
    Train an MLP on MNIST and print the test scores as training progresses.
    """

    if is_test_mode():
        test_period = 200
        minibatch_size = 5
        n_epochs = 0.01
        dataset = get_mnist_dataset(n_training_samples=30, n_test_samples=30)
    else:
        dataset = get_mnist_dataset(n_training_samples=max_training_samples)

    if minibatch_size == 'full':
        minibatch_size = dataset.training_set.n_samples

    # Setup the training and test functions
    classifier = GradientBasedPredictor(
        function = MultiLayerPerceptron(
            layer_sizes=hidden_sizes+[dataset.n_categories],
            input_size = dataset.input_size,
            hidden_activation=hidden_activation,
            output_activation='softmax',
            w_init = normal_w_init(mag = w_init_mag)
            ),
        cost_function=negative_log_likelihood_dangerous,
        optimizer=SimpleGradientDescent(eta = eta)
        ).compile()  # .compile() turns the GradientBasedPredictor, which works with symbolic variables, into a real one that takes and returns arrays.

    # Train and periodically report the test score.
    print 'Running MLP on MNIST Dataset...'
    for i, (_, image_minibatch, label_minibatch) in enumerate(dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_epochs, single_channel = True)):
        if i % test_period == 0:
            epoch = float(i*minibatch_size)/dataset.training_set.n_samples
            training_score = percent_argmax_correct(classifier.predict(dataset.training_set.input), dataset.training_set.target)
            test_score = percent_argmax_correct(classifier.predict(dataset.test_set.input), dataset.test_set.target)
            print 'Epoch %s.  Training: %s%%, Test: %s%%' % (epoch, training_score, test_score)
        classifier.train(image_minibatch, label_minibatch)
    print '...Done.'


EXPERIMENTS = dict()

EXPERIMENTS['regular'] = demo_mnist_mlp

EXPERIMENTS['1000_samples'] = lambda: demo_mnist_mlp(eta = 0.03, minibatch_size=20, max_training_samples=1000, test_period = 100, n_epochs=1000)

EXPERIMENTS['1000_samples-fullbatch'] = lambda: demo_mnist_mlp(eta = 0.03, max_training_samples=1000, minibatch_size='full', test_period = 100, n_epochs=10000)


if __name__ == '__main__':

    which_experiment = '1000_samples'

    logging.getLogger().setLevel(logging.INFO)
    EXPERIMENTS[which_experiment]()
