from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.cost import negative_log_likelihood_dangerous
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.optimization.optimizers import SimpleGradientDescent
from utils.benchmarks.train_and_test import percent_argmax_correct
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
        function = MultiLayerPerceptron.from_init(
            layer_sizes = [dataset.input_size, 100, dataset.n_categories],
            output_activation='softmax',
            w_init = 0.1,
            rng = 3252
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
