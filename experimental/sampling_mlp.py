from plato.tools.cost import negative_log_likelihood
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.compare_predictors import compare_predictors
from utils.datasets.mnist import get_mnist_dataset
from utils.tools.mymath import sqrtspace

__author__ = 'peter'




if __name__ == '__main__':

    # dataset = get_mnist_dataset().process_with(targets_processor=multichannel(OneHotEncoding()))

    results = compare_predictors(
        dataset = get_mnist_dataset(),
        online_predictor_constructors={
            'MLP': lambda: GradientBasedPredictor(
                function = MultiLayerPerceptron(layer_sizes = [500, 10], input_size = 784, output_activation='softmax'),
                cost_function=negative_log_likelihood,
                optimizer=SimpleGradientDescent(eta = 0.1),
                ).compile()
            'Sampling-MLP': lambda:

            },
        evaluation_function='percent_argmax_correct',
        minibatch_size=20,
        test_points=sqrtspace(0, 10, 50)
        )
    plot_learning_curves(results)
