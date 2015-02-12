from scipy.stats.stats import mode
from sklearn.ensemble.forest import RandomForestClassifier, RandomTreesEmbedding
from utils.datasets.mnist import get_mnist_dataset
import numpy as np


def demo_rf_ensemble():

    seed = 1
    x_tr, y_tr, x_ts, y_ts = get_mnist_dataset().xyxy

    # Train a Random forest
    percent_correct = lambda predictions, target: 100.*np.mean(predictions == target)
    rf_classifier = RandomForestClassifier(n_estimators=10, random_state = np.random.RandomState(seed), max_depth = None)
    rf_classifier.fit(x_tr, y_tr)
    rf_predictions = rf_classifier.predict(x_ts)
    rf_score = percent_correct(rf_predictions, y_ts)
    print 'Random Forest score: %s%%' % (rf_score, )

    # Train one tree at a time, take majority vote.  Assert that this has the same result (RandomForestClassifier should
    # be doing this internally)
    rng = np.random.RandomState(seed)
    r_trees = []
    for i in xrange(10):
        tree = RandomForestClassifier(n_estimators=1, random_state = rng, max_depth = None)
        tree.fit(x_tr, y_tr)
        r_trees.append(tree)
    (my_rf_predictions, ), _ = mode(np.array([tree.predict(x_ts) for tree in r_trees]), axis = 0)
    my_rf_score = percent_correct(my_rf_predictions, y_ts)
    assert my_rf_score == rf_score
    print 'My Random Forest score: %s%%' % (my_rf_score, )

    # Now, lets see if we can do better!
    # Using
    # a) Gibbs
    # b) Herded Gibbs
    raise NotImplementedError('Under Construction...')
    rf_embedding = RandomTreesEmbedding(n_estimators=10, random_state = np.random.RandomState(seed))
    rf_embedding.fit(x_tr, y_tr)
    out = rf_embedding.transform(x_tr)
    tree_outputs = rf_classifier.apply(x_ts)


if __name__ == '__main__':

    demo_rf_ensemble()
