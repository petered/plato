from experimental.demo_binary_regression import SamplingPredictor, setup_visualization
from plato.tools.sampling import GibbsRegressor
from plotting.live_plotting import LiveStream
from scipy.stats.stats import mode
from sklearn.ensemble.forest import RandomForestClassifier
from utils.benchmarks.plot_learning_curves import plot_learning_curves
from utils.benchmarks.compare_predictors import assess_incremental_predictor
from utils.benchmarks.train_and_test import percent_correct, percent_argmax_correct
from utils.datasets.datasets import DataSet
from utils.datasets.mnist import get_mnist_dataset
import numpy as np
from utils.tools.mymath import sqrtspace
from utils.tools.processors import RunningAverage, OneHotEncoding


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

    seed = 1
    n_trees = 10
    max_training_samples = 500
    max_test_samples = 500
    plot = False

    mnist_dataset = get_mnist_dataset(n_training_samples=max_training_samples, n_test_samples=max_test_samples)\
        .process_with(inputs_processor = lambda (x, ): (x.reshape(x.shape[0], -1), ))

    # x_tr, y_tr, x_ts, y_ts = mnist_dataset.xyxy

    # Train a bunch of decision trees



    # (mode_tree_output, ), _ = mode(np.array(p_each_tree_test), axis = 0)
    # mode_tree_score = percent_correct(mode_tree_output, y_ts)
    # print 'Mode-tree combination score: %s' % (mode_tree_score, )

    # Now, train our thing instead and see how it compares.
    ensemble_dataset = get_rf_ensemble_dataset(source_dataset = mnist_dataset, n_trees=n_trees, n_classes = 10, seed=seed)

    # label_encoder = OneHotEncoding(n_classes = 10)
    #
    # ensemble_dataset = DataSet.from_xyxy(
    #     training_inputs = merge_and_encode_labels(p_each_tree_training),
    #     training_targets = label_encoder(mnist_dataset.training_set.target),
    #     test_inputs = merge_and_encode_labels(p_each_tree_test),
    #     test_targets = label_encoder(mnist_dataset.test_set.target)
    #     )

    predictor = SamplingPredictor(GibbsRegressor(n_dim_in = n_trees*10, n_dim_out=10, possible_ws=np.linspace(0, 1, 10), n_alpha = 5))

    if plot:
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
            return plot_dict
            # if 'self._phi' in lv:
            #     plot_dict['phi'] = lv['self._phi'].squeeze()

        plotter = LiveStream(get_plotting_vals)

        predictor.train_function.add_callback(plotter.update)

    record = assess_incremental_predictor(
        predictor = predictor,
        dataset = ensemble_dataset.process_with(inputs_processor=lambda (x, ): (x.reshape(x.shape[0], -1, ))),
        evaluation_function=percent_argmax_correct,
        sampling_points=sqrtspace(0, 1000, 50).astype(int),
        accumulation_function='mean',
        which_sets = 'test'
        )

    plot_learning_curves({'Gibbs': record})

    pass

if __name__ == '__main__':

    demo_rf_ensemble()
