import logging
from plato.tools.cost import negative_log_likelihood, percent_correct
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.optimizers import SimpleGradientDescent
from plato.tools.training import SupervisedTrainingFunction, SupervisedTestFunction
from utils.datasets.mnist import get_mnist_dataset

__author__ = 'peter'


def demo_mnist_mlp(test_mode = False):
    """
    Train an MLP on MNIST and print the test scores as training progresses.
    """

    if test_mode:
        test_period = 200
        minibatch_size = 5
        n_epochs = 0.01
    else:
        test_period = 1000
        minibatch_size = 20
        n_epochs = 10

    dataset = get_mnist_dataset()

    # Setup the training and test functions
    classifier = MultiLayerPerceptron(layer_sizes=[500, 10], input_size = 784, hidden_activation='sig', output_activation='lin', w_init_mag=0.01)
    training_cost_function = negative_log_likelihood
    optimizer = SimpleGradientDescent(eta = 0.1)
    training_function = SupervisedTrainingFunction(classifier, training_cost_function, optimizer).compile()
    test_cost_function = percent_correct
    test_function = SupervisedTestFunction(classifier, test_cost_function).compile()

    def report_test(i):
        training_cost = test_function(dataset.training_set.input, dataset.training_set.target)
        print 'Training score at iteration %s: %s' % (i, training_cost)
        test_cost = test_function(dataset.test_set.input, dataset.test_set.target)
        print 'Test score at iteration %s: %s' % (i, test_cost)

    # Train and periodically report the test score.
    print 'Running MLP on MNIST Dataset...'
    for i, (_, image_minibatch, label_minibatch) in enumerate(dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = n_epochs, single_channel = True)):
        if i % test_period == 0:
            report_test(i)
        training_function(image_minibatch, label_minibatch)
    report_test('Final')
    print '...Done.'


if __name__ == '__main__':

    logging.getLogger().setLevel(logging.INFO)
    demo_mnist_mlp()
