from general.checkpoint_counter import CheckPointCounter
from general.should_be_builtins import bad_value
from utils.benchmarks.train_and_test import evaluate_predictor, get_evaluation_function
from collections import OrderedDict
from utils.tools.mymath import sqrtspace
import numpy as np
from utils.tools.processors import RunningAverage
from utils.tools.progress_indicator import ProgressIndicator


def compare_predictors(dataset, online_predictors={}, offline_predictors={}, minibatch_size = 'full',
        evaluation_function = 'mse', test_epochs = sqrtspace(0, 1, 10), report_test_scores = True,
        test_on = 'training+test', test_batch_size = None, accumulators = None):
    """
    Compare a set of predictors by running them on a dataset, and return the learning curves for each predictor.

    :param dataset: A DataSet object
    :param online_predictors: A dict<str:IPredictor> of online predictors.  An online predictor is
        sequentially fed minibatches of data and updates its parameters with each minibatch.
    :param offline_predictors: A dict<str:object> of offline predictors.  Offline predictors obey sklearn's
        Estimator/Predictor interfaces - ie they methods
            estimator = object.fit(data, targets) and
            prediction = object.predict(data)
    :param minibatch_size: Size of the minibatches to use for online predictors.  Can be:
        An int, in which case it represents the minibatch size for all classifiers.
        A dict<str: int>, in which case you can set the minibatch size per-classifier.
        In place of the int, you can put 'all' if you want to train on the whole dataset in each iteration.
    :param test_epochs: Test points to use for online predictors.  Can be:
        A list of integers - in which case the classifier is after seeing this many samples.
        A list of floats - in which case the classifier is tested after seeing this many epochs.
        'always' - In which case a test is performed after every training step
        The final test point determines the end of training.
    :param evaluation_function: Function used to evaluate output of predictors
    :param report_test_scores: Boolean indicating whether you'd like to report results online.
    :return: An OrderedDict<LearningCurveData>
    """

    all_keys = online_predictors.keys()+offline_predictors.keys()
    assert len(all_keys) > 0, 'You have to give at least one predictor.  Is that too much to ask?'
    assert len(all_keys) == len(np.unique(all_keys)), "You have multiple predictors using the same names. Change that."
    type_constructor_dict = OrderedDict(
        [(k, ('offline', offline_predictors[k])) for k in sorted(offline_predictors.keys())] +
        [(k, ('online', online_predictors[k])) for k in sorted(online_predictors.keys())]
        )

    if not isinstance(minibatch_size, dict):
        minibatch_size = {predictor_name: minibatch_size for predictor_name in online_predictors.keys()}
    else:
        assert online_predictors.viewkeys() == minibatch_size.viewkeys()

    if not isinstance(accumulators, dict):
        accumulators = {predictor_name: accumulators for predictor_name in online_predictors.keys()}
    else:
        assert online_predictors.viewkeys() == accumulators.viewkeys()

    test_epochs = np.array(test_epochs)
    # test_epochs_float = test_epochs.dtype == float
    # if test_epochs_float:
    #     test_epochs = (test_epochs * dataset.training_set.n_samples).astype(int)
    if isinstance(evaluation_function, str):
        evaluation_function = get_evaluation_function(evaluation_function)

    records = OrderedDict()

    # Run the offline predictors
    for predictor_name, (predictor_type, predictor) in type_constructor_dict.iteritems():
        print '%s\nRunning predictor %s\n%s' % ('='*20, predictor_name, '-'*20)
        records[predictor_name] = \
            assess_offline_predictor(
                predictor=predictor,
                dataset = dataset,
                evaluation_function = evaluation_function,
                report_test_scores = report_test_scores,
                test_on = test_on,
                test_batch_size = test_batch_size
                ) if predictor_type == 'offline' else \
            assess_online_predictor(
                predictor=predictor,
                dataset = dataset,
                evaluation_function = evaluation_function,
                test_epochs = test_epochs,
                accumulator = accumulators[predictor_name],
                minibatch_size = minibatch_size[predictor_name],
                report_test_scores = report_test_scores,
                test_on = test_on,
                test_batch_size = test_batch_size,
                ) if predictor_type == 'online' else \
            bad_value(predictor_type)

    print 'Done!'

    return records


def dataset_to_testing_sets(dataset, test_on = 'training+test'):
    return \
        {'Training': (dataset.training_set.input, dataset.training_set.target), 'Test': (dataset.test_set.input, dataset.test_set.target)} if test_on == 'training+test' else \
        {'Test': (dataset.test_set.input, dataset.test_set.target)} if test_on == 'test' else \
        {'Training': (dataset.training_set.input, dataset.training_set.target)} if test_on == 'training' else \
        bad_value(test_on)


def assess_offline_predictor(predictor, dataset, evaluation_function, test_on = 'training+test', report_test_scores=True,
        test_batch_size = None):
    """
    Train an offline predictor and return the LearningCurveData (which will not really be a curve,
    but just a point representing the final score.

    :param predictor:  An object with methods fit(X, Y), predict(X)
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: LearningCurveData containing the score on the test sets
    """
    record = LearningCurveData()
    predictor.fit(dataset.training_set.input, dataset.training_set.target)
    testing_sets = dataset_to_testing_sets(dataset, test_on)
    scores = [(k, evaluation_function(process_in_batches(predictor.predict, x, test_batch_size), y)) for k, (x, y) in testing_sets.iteritems()]
    record.add(None, scores)
    if report_test_scores:
        print 'Scores: %s' % (scores, )
    return record


def assess_online_predictor(predictor, dataset, evaluation_function, test_epochs, minibatch_size, test_on = 'training+test',
        accumulator = None, report_test_scores=True, test_batch_size = None):
    """
    Train an online predictor and return the LearningCurveData.

    :param predictor:  An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param test_epochs:
    :param minibatch_size:
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: LearningCurveData containing the score on the test sets
    """

    record = LearningCurveData()

    testing_sets = dataset_to_testing_sets(dataset, test_on)
    if accumulator is None:
        prediction_functions = {k: predictor.predict for k in testing_sets}
    else:
        accum_constructor = {'avg': RunningAverage}[accumulator]
        accumulators = {k: accum_constructor() for k in testing_sets}

        prediction_functions = {k: lambda inp, kp=k: accumulators[kp](predictor.predict(inp)) for k in testing_sets}
        # Bewate the in-loop lambda - but I think we're ok here.

    checker = CheckPointCounter(test_epochs)

    last_n_samples_seen = 0
    for (n_samples_seen, input_minibatch, target_minibatch) in \
            dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = float('inf'), single_channel = True):

        current_epoch = (float(last_n_samples_seen))/dataset.training_set.n_samples
        last_n_samples_seen = n_samples_seen
        time_for_a_test, done = checker.check(current_epoch)
        if time_for_a_test:

            scores = [(k, evaluation_function(process_in_batches(prediction_functions[k], x, test_batch_size), y)) for k, (x, y) in testing_sets.iteritems()]
            if report_test_scores:
                print 'Scores at Epoch %s: %s' % (current_epoch, scores)
            record.add(current_epoch, scores)
            if done:
                break

        predictor.train(input_minibatch, target_minibatch)


    return record


def process_in_batches(func, data, batch_size):
    """
    Sometimes a function requires too much internal memory, so you have to process things in batches.
    """
    if batch_size is None:
        return func(data)

    n_samples = len(data)
    chunks = np.arange(int(np.ceil(float(n_samples)/batch_size))+1)*batch_size
    assert len(chunks)>1
    out = None
    for ix_start, ix_end in zip(chunks[:-1], chunks[1:]):
        x = data[ix_start:ix_end]
        y = func(x)
        if out is None:
            out = np.empty((n_samples, )+y.shape[1:], dtype = y.dtype)
            out[ix_start:ix_end] = y
    return out


class LearningCurveData(object):
    """
    A container for the learning curves resulting from running a predictor
    on a dataset.  Use this object to incrementally write results, and then
    retrieve them as a whole.
    """
    def __init__(self):
        self._times = []
        self._scores = None
        self._latest_score = None

    def add(self, time, scores):
        """
        :param time: Something representing the time at which the record was taken.
        :param scores: A list of 2-tuples of (score_name, score).  It can also be a scalar score.
            Eg: [('training', 0.104), ('test', 0.119)]
        :return:
        """
        if np.isscalar(scores):
            scores = [('Score', scores)]
        elif isinstance(scores, tuple):
            scores = [scores]
        else:
            assert isinstance(scores, list) and all(len(s) == 2 for s in scores)

        self._times.append(time)
        if self._scores is None:
            self._scores = OrderedDict((k, []) for k, _ in scores)
        for k, v in scores:
            self._scores[k].append(v)

    def get_results(self):
        """
        :return: (times, results), where:
            times is a length-N vector indicating the time of each test
            scores is a (length_N, n_scores) array indicating the each score at each time.
        """
        return np.array(self._times), OrderedDict((k, np.array(v)) for k, v in self._scores.iteritems())

    def get_scores(self, which_test_set = None):
        """
        :return: scores for the given test set.
            For an offline predictor, scores'll be float
            For an online predictor, scores'll by a 1-D array representing the score at each test point.
        """
        _, results = self.get_results()

        if which_test_set is None:
            assert len(results)==1, 'You failed to specify which test set to use, which would be fine if there was only ' \
                "one, but there's more than one.  There's %s" % (results.keys(), )
            return results.values()[0]
        else:
            assert which_test_set in results, 'You asked for results for the test set %s, but we only have test sets %s' \
                % (which_test_set, results.keys())
            return results[which_test_set]
