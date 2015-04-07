from plato.tools.cost import negative_log_likelihood, softmax_negative_log_likelihood
from plato.tools.online_prediction.compare_symbolic_predictors import plot_records, CompareOnlinePredictors
from plato.tools.networks import MultiLayerPerceptron, normal_w_init
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent, AdaMax
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.benchmarks.train_and_test import percent_correct
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.mymath import sqrtspace

__author__ = 'peter'


comparisons = lambda: None


"""
Here we run a comparison between SGD and Durk's new pet: AdaMax.  We run them both on MNIST for 50 epochs.
"""
comparisons.adamax_showdown = lambda test_mode = False: compare_predictors(
    dataset = get_mnist_dataset(n_training_samples = 30 if test_mode else None),
    online_predictors = {
        'sgd': GradientBasedPredictor(
            function = MultiLayerPerceptron(layer_sizes=[500, 10], input_size = 784, hidden_activation='sig', output_activation='lin', w_init = normal_w_init(mag = 0.01, seed = 5)),
            cost_function = softmax_negative_log_likelihood,  # *
            optimizer = SimpleGradientDescent(eta = 0.1),
            ).compile(),
        'adamax': GradientBasedPredictor(
            function = MultiLayerPerceptron(layer_sizes=[500, 10], input_size = 784, hidden_activation='sig', output_activation='lin', w_init = normal_w_init(mag = 0.01, seed = 5)),
            cost_function = softmax_negative_log_likelihood,
            optimizer = AdaMax(alpha = 1e-3),
            ).compile(),
        },
    minibatch_size = {
        'sgd': 20,
        'adamax': 20,
        },
    test_epochs = sqrtspace(0, 0.1, 2) if test_mode else sqrtspace(0, 30, 50),
    evaluation_function = percent_correct
    )
# * A more natural-seeming way would be to have output_activation='softmax' and cost_function = negative_log_likelihood
# BUT since negative_log_likelihood has and should have an assert in it to check that inputs are normalized, and theano's
# assert_op has a bug where it doesn't allow gradient calculation (see https://github.com/Theano/Theano/issues/2488), we
# instead have the network output layer be linear and include the softmax in the cost function (where we can safely
# avoid doing an assert, because output of the softmax is normalized by definition.)


if __name__ == '__main__':

    comparison = comparisons.adamax_showdown()
    records = comparison()
    plot_records(records)
