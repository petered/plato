from experimental.demo_binary_regression import SamplingPredictor
from plato.tools.sampling import GibbsRegressor
from scipy.stats.stats import mode
from sklearn.ensemble.forest import RandomForestClassifier
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


def demo_rf_ensemble():

    seed = 1
    n_trees = 10
    mnist_dataset = get_mnist_dataset().process_with(inputs_processor = lambda (x, ): (x.reshape(x.shape[0], -1), ))
    x_tr, y_tr, x_ts, y_ts = mnist_dataset.xyxy

    # Train a bunch of decision trees
    tree_rng = np.random.RandomState(seed)
    trees = [train_tree(x_tr, y_tr, rng = tree_rng) for _ in xrange(n_trees)]
    p_each_tree_training = [t.predict(x_tr) for t in trees]
    p_each_tree_test = [t.predict(x_ts) for t in trees]
    (mode_tree_output, ), _ = mode(np.array(p_each_tree_test), axis = 0)
    mode_tree_score = percent_correct(mode_tree_output, y_ts)
    print 'Mode-tree combination score: %s' % (mode_tree_score, )

    # Now, train our thing instead and see how it compares.
    label_encoder = OneHotEncoding(n_classes = 10)
    # label_decoder = lambda x: np.argmax(x, axis = 1)
    merge_and_encode_labels = lambda labels: np.concatenate([label_encoder(lab) for lab in labels], axis = 1)
    ensemble_dataset = DataSet.from_xyxy(
        training_inputs = merge_and_encode_labels(p_each_tree_training),
        training_targets = mnist_dataset.training_set.target,
        test_inputs = merge_and_encode_labels(p_each_tree_test),
        test_targets = mnist_dataset.test_set.target
        )

    record = assess_incremental_predictor(
        predictor = SamplingPredictor(GibbsRegressor(n_dim_in = n_trees*10, n_dim_out=1, possible_ws=(0, 1))),
        dataset = ensemble_dataset,
        evaluation_function=percent_argmax_correct,
        sampling_points=sqrtspace(0, 1000, 50).astype(int),
        accumulation_function='mean'
        )










if __name__ == '__main__':

    demo_rf_ensemble()
