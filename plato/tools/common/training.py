import time

from artemis.general.checkpoint_counter import CheckPointCounter
from plato.core import create_shared_variable, symbolic
from plato.interfaces.decorators import symbolic_updater, symbolic_simple
from utils.benchmarks.predictor_comparison import LearningCurveData, dataset_to_testing_sets
from artemis.ml.predictors.train_and_test import get_evaluation_function
from artemis.ml.tools.iteration import minibatch_index_generator
from artemis.ml.tools.processors import RunningAverage


__author__ = 'peter'


@symbolic_updater
class SupervisedTrainingFunction(object):

    def __init__(self, classifier, cost_function, optimizer):
        self._classifier = classifier
        self._cost_function = cost_function
        self._optimizer = optimizer

    def __call__(self, data, labels):
        output = self._classifier(data)
        cost = self._cost_function(output, labels)
        updates = self._optimizer(cost=cost, parameters = self._classifier.parameters)
        return updates


@symbolic_simple
class SupervisedTestFunction(object):

    def __init__(self, classifier, cost_function):
        self._classifier = classifier
        self._cost_function = cost_function

    def __call__(self, data, labels):
        output = self._classifier(data)
        cost = self._cost_function(output, labels)
        return cost


def assess_online_symbolic_predictor(predictor, dataset, evaluation_function, test_epochs, minibatch_size, test_on = 'training+test',
        accumulator = None, report_test_scores=True, test_callback = None, add_test_values = True):
    """
    Train an online predictor and return the LearningCurveData.

    :param predictor:  An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param test_epochs: List of epochs to test at.  Eg. [0.5, 1, 2, 4]
    :param minibatch_size: Number of samples per minibatch, or:
        'full' to do full-batch.
        'stretch': to stretch the size of each batch so that we make just one call to "train" between each test.  Use
            this, for instance, if your predictor trains on one sample at a time in sequence anyway.
    :param test_on: 'test' to test only on the test set, or 'training+test' to test on both.
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :param test_callback: A callback which takes the predictor, and is called every time a test
        is done.  This can be useful for plotting/debugging the state.
    :return: LearningCurveData containing the score on the test sets
    """

    record = LearningCurveData()

    testing_sets = dataset_to_testing_sets(dataset, test_on)
    if accumulator is None:
        prediction_functions = {k: predictor.predict for k in testing_sets}
    else:
        accum_constructor = {'avg': RunningAverage}[accumulator] \
            if isinstance(accumulator, str) else accumulator
        accumulators = {k: accum_constructor() for k in testing_sets}
        prediction_functions = {k: lambda inp, kp=k: accumulators[kp](predictor.predict(inp)) for k in testing_sets}
        # Bewate the in-loop lambda - but I think we're ok here.

    if isinstance(evaluation_function, str):
        evaluation_function = get_evaluation_function(evaluation_function)

    @symbolic
    def train(indices):
        x_tr = create_shared_variable(dataset.training_set.input)
        y_tr = create_shared_variable(dataset.training_set.target)
        predictor.train(x_tr[indices], y_tr[indices])

    @symbolic
    def test_on_test_set():
        x_ts = create_shared_variable(dataset.test_set.input)
        outputs = predictor.predict(x_ts)
        return outputs

    @symbolic
    def test_on_training_set():
        x_tr = create_shared_variable(dataset.training_set.input)
        outputs = predictor.predict(x_tr)
        return outputs

    train_fcn = train.compile(add_test_values=add_test_values)
    prediction_functions = {
        'Training': test_on_training_set.compile(add_test_values=add_test_values),
        'Test': test_on_test_set.compile(add_test_values=add_test_values),
        }

    def do_test(current_epoch):
        scores = [(k, evaluation_function(prediction_functions[k](), y)) for k, (x, y) in testing_sets.iteritems()]
        if report_test_scores:
            print 'Scores at Epoch %s: %s, (after %ss)' % (current_epoch, ', '.join('%s: %.3f' % (set_name, score) for set_name, score in scores), time.time()-start_time)
        record.add(current_epoch, scores)
        if test_callback is not None:
            record.add(current_epoch, ('callback', test_callback(predictor)))

    checker = CheckPointCounter(test_epochs)
    last_n_samples_seen = 0
    start_time = time.time()

    for indices in minibatch_index_generator(n_samples=dataset.training_set.n_samples, minibatch_size=minibatch_size, n_epochs=float('inf'), slice_when_possible=False):
        current_epoch = (float(last_n_samples_seen))/dataset.training_set.n_samples
        last_n_samples_seen += minibatch_size
        time_for_a_test, done = checker.check(current_epoch)
        if time_for_a_test:
            do_test(current_epoch)
        if done:
            break
        train_fcn(indices)

    return record
    #         time_for_a_test, done = checker.check(current_epoch)
    #     for (n_samples_seen, input_minibatch, target_minibatch) in \
    #             dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = float('inf'), single_channel = True):
    #         current_epoch = (float(last_n_samples_seen))/dataset.training_set.n_samples
    #         last_n_samples_seen = n_samples_seen
    #         time_for_a_test, done = checker.check(current_epoch)
    #         if time_for_a_test:
    #             do_test(current_epoch)
    #         if done:
    #             break
    #         predictor.train(input_minibatch, target_minibatch)
    #
    # return record
