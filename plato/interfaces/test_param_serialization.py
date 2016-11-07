import pickle
from pickle import PicklingError

import pytest
import numpy as np

from plato.interfaces.param_serialzation import dumps_params, loads_params
from plato.tools.mlp.mlp import MultiLayerPerceptron
from plato.tools.optimization.cost import negative_log_likelihood_dangerous
from plato.tools.common.online_predictors import GradientBasedPredictor
from plato.tools.optimization.optimizers import SimpleGradientDescent
from artemis.ml.predictors.train_and_test import train_online_predictor, evaluate_predictor, percent_argmax_correct
from artemis.ml.datasets.synthetic_clusters import get_synthetic_clusters_dataset


__author__ = 'peter'


"""
This file shows some approaches to serializing predictors.  There are 3 ways to go about it as I see:

1) Just serialize parameters (see test_param_serialization)
    Pros:
    - Your files contain only native python/numpy objects, so your pickes will never break (no PicklingErrors)
    Cons:
    - You cannot instatiate an object from the pickle file - you must keep the code to instatiate it, and then
      load the params in.
2) Just pickle predictors (see test_predictor+_pickling)
    Pros:
    - Instatiated predictor is restored from pickle.
    - No additional work required if your predictor can just serialize right off the bat.
    Cons:
    - Can break if code is moved or changed. (Code is always moved or changed).  In this case, we have
      to track version, etc.
    - We will often have to deal with __getstate__, __setstate__ and special pickling method for dealing
      with unpicklable fields like lambda functions.
    - When moving stuff, you have to leave behind "dummy" classes if you want to reload old pickles that
      were based on those classes.
3) Pickle in standardized format (not shown) have some standardized format in which to save e.g. an MLP.  And convert
   your object to this format (so that it only has native numpy/python objects in it)
    Pros:
    - Your serialized object is code-independent (no PicklingErrors, other people using different code can use it, if
      they make the mapping function from their MLP to the format and back.
    - You can instantiate a new object from file
    Cons:
    - You have to maintain the mapping of your object to the standard format and back.
"""


def test_param_serialization():
    """
    Pros -
    :return:
    """

    dataset = get_synthetic_clusters_dataset()

    predictor_constructor = lambda: GradientBasedPredictor(
        function = MultiLayerPerceptron.from_init(
            layer_sizes = [dataset.input_shape[0], 100, dataset.n_categories],
            output_activation='softmax',
            w_init = lambda n_in, n_out, rng = np.random.RandomState(3252): 0.1*rng.randn(n_in, n_out)
            ),
        cost_function=negative_log_likelihood_dangerous,
        optimizer=SimpleGradientDescent(eta = 0.1),
        ).compile()

    evaluate = lambda pred: evaluate_predictor(pred, dataset.test_set, percent_argmax_correct)

    # Train up predictor and save params
    predictor = predictor_constructor()
    pre_training_score = evaluate(predictor)
    assert pre_training_score < 35
    train_online_predictor(predictor, dataset.training_set, minibatch_size=20, n_epochs=3)
    post_training_score = evaluate(predictor)
    assert post_training_score > 95
    trained_param_string = dumps_params(predictor)

    # Instantiate new predictor and load params
    new_predictor = predictor_constructor()
    new_pre_training_score = evaluate(new_predictor)
    assert new_pre_training_score < 35
    loads_params(new_predictor, trained_param_string)
    loaded_score = evaluate(new_predictor)
    assert loaded_score == post_training_score > 95


def test_predictor_pickling():

    dataset = get_synthetic_clusters_dataset()

    predictor_constructor = lambda: GradientBasedPredictor(
        function = MultiLayerPerceptron.from_init(
            layer_sizes = [dataset.input_shape[0], 100, dataset.n_categories],
            output_activation='softmax',
            w_init = lambda n_in, n_out, rng = np.random.RandomState(3252): 0.1*rng.randn(n_in, n_out)
            ),
        cost_function=negative_log_likelihood_dangerous,
        optimizer=SimpleGradientDescent(eta = 0.1),
        ).compile()

    evaluate = lambda pred: evaluate_predictor(pred, dataset.test_set, percent_argmax_correct)

    # Train up predictor and save params
    predictor = predictor_constructor()
    pre_training_score = evaluate(predictor)
    assert pre_training_score < 35
    train_online_predictor(predictor, dataset.training_set, minibatch_size=20, n_epochs=3)
    post_training_score = evaluate(predictor)
    assert post_training_score > 95

    with pytest.raises(PicklingError):
        # TODO: Fix the PicklingError
        trained_predictor_string = pickle.dumps(predictor)

        # Instantiate new predictor and load params
        new_predictor = pickle.loads(trained_predictor_string)
        loaded_score = evaluate(new_predictor)
        assert loaded_score == post_training_score > 95


if __name__ == '__main__':
    test_predictor_pickling()
    test_param_serialization()
