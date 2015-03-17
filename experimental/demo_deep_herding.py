from experimental.sampling_mlp import GibbsSamplingMLP
from plato.tools.cost import negative_log_likelihood
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.predictor_comparison import compare_predictors
from utils.datasets.mnist import get_mnist_dataset
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from utils.tools.mymath import sqrtspace
import numpy as np

def demo_mnist_herding():

    results = compare_predictors(
        # dataset = get_mnist_dataset(),
        dataset = get_synthetic_clusters_dataset(),
        online_predictors={
            # 'MLP': GradientBasedPredictor(
            #     function = MultiLayerPerceptron(layer_sizes = [10], input_size = 20, output_activation='lin', w_init = lambda n_in, n_out: 0.1*np.random.randn(n_in, n_out)),
            #     cost_function=negative_log_likelihood,
            #     optimizer=SimpleGradientDescent(eta = 0.1),
            #     ).compile(),
            'Gibbs-MLP': GibbsSamplingMLP(
                layer_sizes = [10],
                input_size = 20,
                possible_ws=(-1, 0, 1),
                output_activation='softmax'
                ).compile(mode = 'debug'),
            },
        evaluation_function='percent_argmax_correct',
        minibatch_size='full',
        test_epochs=sqrtspace(0, 20, 20)
        )
    plot_learning_curves(results)


if __name__ == '__main__':

    demo_mnist_herding()
    # dataset = get_mnist_dataset().process_with(targets_processor=multichannel(OneHotEncoding()))
