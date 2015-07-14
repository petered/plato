from plato.tools.cost import negative_log_likelihood_dangerous
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import GradientBasedPredictor
from plato.tools.optimizers import SimpleGradientDescent
import numpy as np
from utils.benchmarks.train_and_test import evaluate_predictor, percent_argmax_correct
from utils.bureaucracy import zip_minibatch_iterate
from utils.datasets.synthetic_clusters import get_synthetic_clusters_dataset

__author__ = 'peter'


def test_symbolic_predicors():
    """
    This test is meant to serves as both a test and tutorial for how to use a symbolic predictor.
    It shows how to construct a symbolic predictor using a function, cost function, and optimizer.
    It then trains this predictor on a synthetic toy dataset and demonstrates that it has learned.
    """

    dataset = get_synthetic_clusters_dataset()

    symbolic_predictor = GradientBasedPredictor(
        function = MultiLayerPerceptron(
            layer_sizes = [100, dataset.n_categories],
            input_size = dataset.input_size,
            output_activation='softmax',
            w_init = lambda n_in, n_out, rng = np.random.RandomState(3252): 0.1*rng.randn(n_in, n_out)
            ),
        cost_function=negative_log_likelihood_dangerous,
        optimizer=SimpleGradientDescent(eta = 0.1),
        )

    predictor = symbolic_predictor.compile()
    # .compile() turns the symbolic predictor into an IPredictor object, which can be called with numpy arrays.

    init_score = percent_argmax_correct(predictor.predict(dataset.test_set.input), dataset.test_set.target)
    for x_m, y_m in zip_minibatch_iterate([dataset.training_set.input, dataset.training_set.target], minibatch_size=10, n_epochs=20):
        predictor.train(x_m, y_m)
    final_score = percent_argmax_correct(predictor.predict(dataset.test_set.input), dataset.test_set.target)

    print 'Initial score: %s%%.  Final score: %s%%' % (init_score, final_score)
    assert init_score < 30
    assert final_score > 98


if __name__ == '__main__':
    test_symbolic_predicors()
