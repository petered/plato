from experimental.demo_binary_regression import SamplingPredictor
from plato.tools.sampling import GibbsRegressor, HerdedGibbsRegressor
from plotting.live_plotting import LiveStream
from scipy.stats.stats import mode
from sklearn.ensemble.forest import RandomForestClassifier
from utils.benchmarks.compare_predictors import compare_predictors
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.datasets.datasets import DataSet
from utils.datasets.mnist import get_mnist_dataset
import numpy as np
from utils.predictors.mock_predictor import MockPredictor
from utils.tools.mymath import sqrtspace
from utils.tools.processors import OneHotEncoding


def train_and_predict_random_forest(x_tr, y_tr, x_ts, n_trees, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    rf_classifier = RandomForestClassifier(n_estimators=n_trees, random_state = rng, max_depth = None)
    rf_classifier.fit(x_tr, y_tr)
    rf_predictions = rf_classifier.predict(x_ts)
    return rf_predictions


def train_tree(x_tr, y_tr, rng=None):
    tree = RandomForestClassifier(n_estimators=1, random_state = rng, max_depth = None)
    tree.fit(x_tr, y_tr)
    return tree


def train_and_predict_decision_tree(x_tr, y_tr, x_ts, rng=None):
    tree = train_tree(x_tr, y_tr, rng)
    tree_predictions = tree.predict(x_ts)
    return tree_predictions


def get_rf_ensemble_dataset(source_dataset, n_trees, n_classes = None, seed=None):

    if n_classes is None:
        n_classes = np.max(source_dataset.training_set.target)

    x_tr, y_tr, x_ts, y_ts = source_dataset.xyxy

    tree_rng = np.random.RandomState(seed)
    trees = [train_tree(x_tr, y_tr, rng = tree_rng) for _ in xrange(n_trees)]
    p_each_tree_training = [t.predict(x_tr) for t in trees]
    p_each_tree_test = [t.predict(x_ts) for t in trees]

    label_encoder = OneHotEncoding(n_classes = n_classes)

    merge_and_encode_labels = lambda labels: np.concatenate([label_encoder(lab)[:, None, :] for lab in labels], axis = 1)

    ensemble_dataset = DataSet.from_xyxy(
        training_inputs = merge_and_encode_labels(p_each_tree_training),
        training_targets = label_encoder(source_dataset.training_set.target),
        test_inputs = merge_and_encode_labels(p_each_tree_test),
        test_targets = label_encoder(source_dataset.test_set.target)
        )

    return ensemble_dataset


def demo_rf_ensemble():
    # TODO: Remove redundancy with demo_binary_regression

    seed = 1
    n_trees = 10
    max_training_samples = 100
    max_test_samples = 1000
    plot = False
    n_steps = 2000
    n_test_points = 50
    which_dataset = 'ensemble'
    n_classes = 10  # Don't change this

    mnist_dataset = get_mnist_dataset(n_training_samples=max_training_samples, n_test_samples=max_test_samples)\
        .process_with(inputs_processor = lambda (x, ): (x.reshape(x.shape[0], -1), ))

    if which_dataset == 'ensemble':
        # Now, train our thing instead and see how it compares.
        dataset = get_rf_ensemble_dataset(source_dataset = mnist_dataset, n_trees=n_trees, n_classes = 10, seed=seed)
    else:
        raise Exception()

    def get_mode_prediction(x):
        x = x.reshape(len(x), n_trees, n_classes)
        (mode_tree_output, ), _ = mode(np.argmax(x, axis = 2).T, axis = 0)
        binary_output = OneHotEncoding(n_classes)(mode_tree_output)
        return binary_output

    def setup_visualization(predictor):
        if isinstance(predictor, SamplingPredictor):
            predictor.train_function.set_debug_variables('locals+class')
            predictor.predict_function.set_debug_variables('locals')

            def get_plotting_vals():
                training_locals = predictor.train_function.get_debug_values()
                test_locals = predictor.predict_function.get_debug_values()
                plot_dict = {
                    'alpha': training_locals['self._alpha'],
                    'w': training_locals['self._w'],
                    'p_wa': training_locals['p_wa'].squeeze(),
                    'y': training_locals['y'][:50],
                    'p_y': test_locals['p_y'][:50]
                    }
                if 'self._phi' in training_locals:
                    plot_dict['phi'] = training_locals['self._phi'].squeeze()
                return plot_dict

            plotter = LiveStream(get_plotting_vals)
            predictor.train_function.add_callback(plotter.update)

    learning_curves = compare_predictors(
        dataset = dataset.process_with(inputs_processor=lambda (x, ): (x.reshape(x.shape[0], -1, ))),
        offline_predictor_constructors={
            'Mode-Combination': lambda: MockPredictor(get_mode_prediction),
            },
        incremental_predictor_constructors = {
            'gibbs': lambda: SamplingPredictor(GibbsRegressor(n_dim_in = n_trees*n_classes, n_dim_out=n_classes,
                possible_ws=np.linspace(0, 1, 5),
                n_alpha = 5
                )),
            'herded-gibbs': lambda: SamplingPredictor(HerdedGibbsRegressor(n_dim_in = n_trees*n_classes, n_dim_out=n_classes,
                possible_ws=np.linspace(0, 1, 5),
                n_alpha = 5
                )),
            },
        test_points = sqrtspace(0, n_steps, n_test_points),
        evaluation_function = 'percent_argmax_correct',
        report_test_scores=False,
        on_construction_callback=setup_visualization if plot else None
        )

    plot_learning_curves(learning_curves)


if __name__ == '__main__':

    demo_rf_ensemble()
