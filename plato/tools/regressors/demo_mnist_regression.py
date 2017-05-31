from artemis.experiments.experiment_record import ExperimentFunction
from artemis.experiments.ui import browse_experiments
from artemis.general.test_mode import is_test_mode
from artemis.ml.datasets.mnist import get_mnist_dataset
from artemis.ml.predictors.train_and_test import train_and_test_online_predictor
from plato.tools.optimization.optimizers import get_named_optimizer
from plato.tools.regressors.online_regressor import OnlineRegressor
import numpy as np
__author__ = 'peter'


@ExperimentFunction(one_liner_results=lambda info_score_pair_sequence: info_score_pair_sequence.get_oneliner(), is_root=True)
def demo_mnist_online_regression(
        minibatch_size = 10,
        learning_rate = 0.1,
        optimizer = 'sgd',
        regressor_type = 'multinomial',
        n_epochs = 20,
        n_test_points = 30,
        max_training_samples = None,
        include_biases = True,
        ):
    """
    Train an MLP on MNIST and print the test scores as training progresses.
    """

    if is_test_mode():
        n_test_points = 3
        minibatch_size = 5
        n_epochs = 0.01
        dataset = get_mnist_dataset(n_training_samples=30, n_test_samples=30, flat = True)
    else:
        dataset = get_mnist_dataset(n_training_samples=max_training_samples, flat = True)

    assert regressor_type in ('multinomial', 'logistic', 'linear')

    n_outputs = dataset.n_categories
    if regressor_type in ('logistic', 'linear'):
        dataset = dataset.to_onehot()

    predictor = OnlineRegressor(
        input_size = dataset.input_size,
        output_size = n_outputs,
        regressor_type = regressor_type,
        optimizer=get_named_optimizer(name = optimizer, learning_rate=learning_rate),
        include_biases = include_biases
        )

    # Train and periodically report the test score.
    results = train_and_test_online_predictor(
        dataset=dataset,
        train_fcn=predictor.train.compile(),
        predict_fcn = predictor.predict.compile(),
        minibatch_size=minibatch_size,
        n_epochs=n_epochs,
        test_epochs=np.linspace(0, n_epochs, n_test_points),
        )
    return results


demo_mnist_online_regression.add_variant(regressor_type='multinomial')  # Gets to about 92.5
demo_mnist_online_regression.add_variant(regressor_type='multinomial', learning_rate = 0.01)
demo_mnist_online_regression.add_variant(regressor_type='multinomial', learning_rate = 0.01, minibatch_size=1)
demo_mnist_online_regression.add_variant(regressor_type='multinomial', learning_rate = 0.001, n_epochs=50)
demo_mnist_online_regression.add_variant(regressor_type='multinomial', include_biases=False)  # Also gets to about 92.5.  So at least for MNIST you don't really need a bias term.
demo_mnist_online_regression.add_variant(regressor_type='linear', learning_rate=0.01)  # Requires a lower learning rate for stability, and then only makes it to around 86%
demo_mnist_online_regression.add_variant(regressor_type='logistic', learning_rate=0.01)  # Gets just over 92%


if __name__ == '__main__':

    browse_experiments()
