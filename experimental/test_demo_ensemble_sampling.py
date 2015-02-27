from experimental.old_demo_ensemble_sampling import train_and_predict_decision_tree, train_and_predict_random_forest
from utils.datasets.mnist import get_mnist_dataset
from scipy.stats.stats import mode
import numpy as np

__author__ = 'peter'


def test_random_forest_averaging():
    """
    This just checks that sklearns RandomForest is doing what we thing it's doing.
    """

    seed = 56
    n_trees = 10
    x_tr, y_tr, x_ts, y_ts = get_mnist_dataset(n_training_samples=100, n_test_samples=100).xyxy
    p_rf = train_and_predict_random_forest(x_tr, y_tr, x_ts, n_trees = n_trees, rng = np.random.RandomState(seed))
    tree_rng = np.random.RandomState(seed)
    p_each_tree = [train_and_predict_decision_tree(x_tr, y_tr, x_ts, rng = tree_rng) for _ in xrange(n_trees)]
    (p_trees, ), _ = mode(np.array(p_each_tree), axis = 0)
    assert np.array_equal(p_rf, p_trees)

if __name__ == '__main__':
    test_random_forest_averaging()
