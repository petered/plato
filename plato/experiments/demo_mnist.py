from plato.tools.cost import NegativeLogLikelihood, PercentCorrect
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.optimizers import SimpleGradientDescent
from plato.tools.training import SupervisedTrainingFunction, SupervisedTestFunction
from utils.datasets.mnist import get_mnist_dataset

__author__ = 'peter'


def demo_mnist():
    """
    Train an MLP on MNIST and print the test scores as training progresses
    """

    test_period = 100

    dataset = get_mnist_dataset()

    # Setup the training and test functions
    classifier = MultiLayerPerceptron(layer_sizes=[500, 10], input_size = [784], hidden_activation='tanh', output_activation='softmax', w_init_mag=0.1)
    training_cost_function = NegativeLogLikelihood()
    optimizer = SimpleGradientDescent(eta = 0.1)
    training_function = SupervisedTrainingFunction(classifier, training_cost_function, optimizer)
    test_cost_function = PercentCorrect()
    test_function = SupervisedTestFunction(classifier, test_cost_function)

    def report_test():
        test_cost = test_function(dataset.test_set.input, dataset.test_set.target)
        print 'Test cost at iteration %s: %s' % (i, test_cost)

    # Train and periodically report the test score.
    for i, (image_minibatch, label_minibatch) in enumerate(dataset.training_set.batch_iterator(minibatch_size = 10, epochs = 1, single_channel = True)):
        if i % test_period == 0:
            report_test()
        training_function(image_minibatch, label_minibatch)
    report_test()


if __name__ == '__main__':

    demo_mnist()
