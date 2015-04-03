from plato.interfaces.param_serialzation import dumps_params, loads_params
from plato.tools.cost import negative_log_likelihood_dangerous
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent
from utils.benchmarks.train_and_test import train_online_predictor, evaluate_predictor, percent_argmax_correct
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset
import numpy as np

__author__ = 'peter'


def test_param_serialization():

    dataset = get_synthetic_clusters_dataset()

    predictor_constructor = lambda: GradientBasedPredictor(
        function = MultiLayerPerceptron(
            layer_sizes = [100, dataset.n_categories],
            input_size = dataset.input_shape[0],
            output_activation='softmax',
            w_init = lambda n_in, n_out, rng = np.random.RandomState(3252): 0.1*rng.randn(n_in, n_out)
            ),
        cost_function=negative_log_likelihood_dangerous,
        optimizer=SimpleGradientDescent(eta = 0.1),
        ).compile()

    evaluate = lambda pred: evaluate_predictor(pred, dataset.test_set, percent_argmax_correct)

    # Train up predictor and save params
    predictor = predictor_constructor()
    pre_training_score = evaluate(predictor)
    assert pre_training_score < 35
    train_online_predictor(predictor, dataset.training_set, minibatch_size=20, n_epochs=3)
    post_training_score = evaluate(predictor)
    assert post_training_score > 95
    trained_param_string = dumps_params(predictor)

    # Instantiate new predictor and load params
    new_predictor = predictor_constructor()
    new_pre_training_score = evaluate(new_predictor)
    assert new_pre_training_score < 35
    loads_params(new_predictor, trained_param_string)
    loaded_score = evaluate(new_predictor)
    assert loaded_score == post_training_score > 95


if __name__ == '__main__':
    test_param_serialization()
