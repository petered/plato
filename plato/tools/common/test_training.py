from plato.tools.common.training import assess_online_symbolic_predictor
from plato.tools.optimization.optimizers import GradientDescent
from plato.tools.regressors.online_regressor import OnlineRegressor
from artemis.ml.datasets.synthetic_clusters import get_synthetic_clusters_dataset

__author__ = 'peter'


def test_assess_online_symbolic_predictor():


    dataset = get_synthetic_clusters_dataset(dtype = 'float32')

    predictor = OnlineRegressor(
                input_size = dataset.input_size,
                output_size=dataset.n_categories,
                optimizer=GradientDescent(eta = 0.01),
                regressor_type = 'multinomial'
                )

    record = assess_online_symbolic_predictor(
        predictor = predictor,
        dataset = dataset,
        evaluation_function='percent_argmax_correct',
        test_epochs=[0, 1, 2],
        minibatch_size=20,
    )

    scores = record.get_scores('Test')
    assert scores[0] <= 40
    assert scores[-1] >= 99
    scores = record.get_scores('Training')
    assert scores[0] <= 40
    assert scores[-1] >= 99

if __name__ == '__main__':
    test_assess_online_symbolic_predictor()