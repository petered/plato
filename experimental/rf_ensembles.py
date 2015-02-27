from utils.datasets.datasets import DataSet
from utils.datasets.mnist import get_mnist_dataset
from utils.predictors.i_predictor import IPredictor
import numpy as np
from scipy.stats.stats import mode
from sklearn.ensemble.forest import RandomForestClassifier
from utils.tools.processors import OneHotEncoding

__author__ = 'peter'


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


def get_mnist_rf_ensemble_dataset(max_training_samples, max_test_samples,
        n_trees, seed = None):

    mnist_dataset = get_mnist_dataset(n_training_samples=max_training_samples, n_test_samples=max_test_samples)\
        .process_with(inputs_processor = lambda (x, ): (x.reshape(x.shape[0], -1), ))

    mnist_rf_ensemble_dataset = get_rf_ensemble_dataset(source_dataset = mnist_dataset, n_trees=n_trees, n_classes = 10, seed=seed)
    return mnist_rf_ensemble_dataset


def get_rf_ensemble_dataset(source_dataset, n_trees, n_classes = None, seed=None):

    if n_classes is None:
        n_classes = np.max(source_dataset.training_set.target)

    x_tr, y_tr, x_ts, y_ts = source_dataset.xyxy

    tree_rng = np.random.RandomState(seed)
    trees = [train_tree(x_tr, y_tr, rng = tree_rng) for _ in xrange(n_trees)]
    p_each_tree_training = [t.predict(x_tr) for t in trees]
    p_each_tree_test = [t.predict(x_ts) for t in trees]

    label_encoder = OneHotEncoding(n_classes = n_classes)

    merge_and_encode_labels = lambda labels: np.concatenate([label_encoder(lab)[:, :] for lab in labels], axis = 1)

    ensemble_dataset = DataSet.from_xyxy(
        training_inputs = merge_and_encode_labels(p_each_tree_training),
        training_targets = label_encoder(source_dataset.training_set.target),
        test_inputs = merge_and_encode_labels(p_each_tree_test),
        test_targets = label_encoder(source_dataset.test_set.target)
        )

    return ensemble_dataset



class MockModePredictor(IPredictor):

    def __init__(self, n_classes):
        self._n_classes = n_classes

    def train(self, input_data, target_data):
        pass  # This is a mock predictor used for ensembles - it just takes the mode of the data

    def predict(self, input_data):
        """
        input_data is an (n_samples, n_trees*n_classes) array of one-hot encoded tree-predictions
        """
        x = input_data.reshape(len(input_data), -1, self._n_classes)
        (mode_tree_output, ), _ = mode(np.argmax(x, axis = 2).T, axis = 0)
        binary_output = OneHotEncoding(self._n_classes)(mode_tree_output)
        return binary_output
