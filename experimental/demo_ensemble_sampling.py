from scipy.stats.stats import mode
from sklearn.ensemble.forest import RandomForestClassifier
from utils.benchmarks.train_and_test import percent_correct
from utils.datasets.mnist import get_mnist_dataset
import numpy as np


def train_and_predict_random_forest(x_tr, y_tr, x_ts, n_trees, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    rf_classifier = RandomForestClassifier(n_estimators=n_trees, random_state = rng, max_depth = None)
    rf_classifier.fit(x_tr, y_tr)
    rf_predictions = rf_classifier.predict(x_ts)
    return rf_predictions


def train_and_predict_decision_tree(x_tr, y_tr, x_ts, rng=None):
    tree = RandomForestClassifier(n_estimators=1, random_state = rng, max_depth = None)
    tree.fit(x_tr, y_tr)
    tree_predictions = tree.predict(x_ts)
    return tree_predictions


def demo_rf_ensemble():

    seed = 1
    n_trees = 10
    x_tr, y_tr, x_ts, y_ts = get_mnist_dataset().xyxy

    # Train a bunch of decision trees
    tree_rng = np.random.RandomState(seed)
    p_each_tree = [train_and_predict_decision_tree(x_tr, y_tr, x_ts, rng = tree_rng) for _ in xrange(n_trees)]
    (mode_tree_output, ), _ = mode(np.array(p_each_tree), axis = 0)
    mode_tree_score = percent_correct(mode_tree_output, y_ts)
    print 'Mode-tree combination score: %s' % (mode_tree_score, )


if __name__ == '__main__':

    demo_rf_ensemble()
