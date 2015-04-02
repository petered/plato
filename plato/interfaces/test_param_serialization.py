from plato.tools.cost import negative_log_likelihood_dangerous
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent
from utils.bureaucracy import multichannel
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset
from utils.tools.processors import OneHotEncoding

__author__ = 'peter'


def test_param_serialization():

    dataset = get_synthetic_clusters_dataset().process_with(targets_processor=multichannel(OneHotEncoding()))

    symbolic_predictor = GradientBasedPredictor(
        function = MultiLayerPerceptron(
            layer_sizes = [100, dataset.n_categories],
            input_size = dataset.target_shape[0],
            output_activation='softmax',
            w_init = lambda n_in, n_out, rng = np.random.RandomState(3252): 0.1*rng.randn(n_in, n_out)
            ),
        cost_function=negative_log_likelihood_dangerous,
        optimizer=SimpleGradientDescent(eta = 0.1),
        )

    predictor = symbolic_predictor.compile()




if __name__ == '__main__':
    test_something()
