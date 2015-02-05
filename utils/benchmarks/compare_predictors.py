from utils.benchmarks.train_and_test import evaluate_predictor, get_evaluation_function
from collections import OrderedDict
from utils.tools.mymath import sqrtspace
import numpy as np
from utils.tools.processors import RunningAverage
from utils.tools.progress_indicator import ProgressIndicator


def compare_predictors(dataset, online_predictor_constructors = {}, offline_predictor_constuctors = {}, incremental_predictor_constructors = {},
        minibatch_size = 1, test_points = sqrtspace(0, 1, 10), evaluation_function = 'mse', report_test_scores = True):
    """
    :param dataset: A DataSet object
    :param online_predictor_constructors: A dict<str:function> where the functions construct IClassifier objects
    :param minibatch_size: Size of the minibatches to use.  Can be:
        An int, in which case it represents the minibatch size for all classifiers.
        A dict<str: int>, in which case you can set the minibatch size per-classifier.
        In place of the int, you can put 'all' if you want to train on the whole dataset in each iteration.
    :param test_points: Test points to use.  Can be:
        A list of integers - in which case the classifier is after seeing this many samples.
        A list of floats - in which case the classifier is tested after seeing this many epochs.
        'always' - In which case a test is performed after every training step
        The final test point determines the end of training.
    :param evaluation_function: Function used to evaluate output of predictors
    :param report_test_scores: Boolean indicating whether you'd like to report results online.
    :return: An OrderedDict<PredictionResult>
    """

    assert not (online_predictor_constructors is None and offline_predictor_constuctors is None and incremental_predictor_constructors is None), \
        'You have to give some predictors.'

    all_keys = online_predictor_constructors.keys()+offline_predictor_constuctors.keys()+incremental_predictor_constructors.keys()
    assert len(all_keys)==len(np.unique(all_keys)), "You have multiple predictors using the same names. Change that."

    if isinstance(minibatch_size, int):
        minibatch_size = {predictor_name: minibatch_size for predictor_name in online_predictor_constructors.keys()}
    else:
        assert online_predictor_constructors.viewkeys() == minibatch_size.viewkeys()
    test_points = np.array(test_points)
    test_points_float = test_points.dtype == float
    if test_points_float:
        test_points = (test_points * dataset.training_set.n_samples).astype(int)
    if isinstance(evaluation_function, str):
        evaluation_function = get_evaluation_function(evaluation_function)

    records = OrderedDict()

    # Run the offline predictors
    for predictor_name, predictor_constructor in offline_predictor_constuctors.iteritems():
        print '%s\nRunning predictor %s\n%s' % ('='*20, predictor_name, '-'*20)
        records[predictor_name] = assess_offline_predictor(
            predictor=predictor_constructor(),
            dataset = dataset,
            evaluation_function = evaluation_function,
            report_test_scores = report_test_scores
            )

    # Run the online predictors
    for predictor_name, predictor_constructor in online_predictor_constructors.iteritems():
        print '%s\nRunning predictor %s\n%s' % ('='*20, predictor_name, '-'*20)
        records[predictor_name] = assess_online_predictor(
            predictor=predictor_constructor(),
            dataset = dataset,
            evaluation_function = evaluation_function,
            test_points = test_points,
            minibatch_size = minibatch_size[predictor_name],
            report_test_scores = report_test_scores
            )

    # Run the incremental predictors
    for predictor_name, predictor_constructor in incremental_predictor_constructors.iteritems():
        print '%s\nRunning predictor %s\n%s' % ('='*20, predictor_name, '-'*20)
        records[predictor_name] = assess_incremental_predictor(
            predictor=predictor_constructor(),
            dataset = dataset,
            evaluation_function = evaluation_function,
            sampling_points = np.sort(np.unique(np.ceil(test_points/dataset.training_set.n_samples).astype(int))),
            accumulation_function='mean',
            report_test_scores = report_test_scores
            )
        # print 'Scores for %s: %s' % (predictor_name, records[predictor_name].get_results()[1]['Test'])

    print 'Done!'

    return records


def assess_offline_predictor(predictor, dataset, evaluation_function, report_test_scores=True):
    """

    :param predictor:  An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: A PredictionResult containing the score on the test set
    """
    record = PredictionResult()
    predictor.train(dataset.training_set.input, dataset.training_set.target)
    training_cost = evaluate_predictor(predictor, test_set = dataset.training_set, evaluation_function=evaluation_function)
    test_cost = evaluate_predictor(predictor, test_set = dataset.test_set, evaluation_function=evaluation_function)
    record.add(None, [('Test', test_cost), ('Training', training_cost)])
    if report_test_scores:
        print 'Training score: %s' % (training_cost, )
        print 'Test score: %s' % (test_cost, )
    return record


def assess_online_predictor(predictor, dataset, evaluation_function, test_points, minibatch_size, report_test_scores=True):
    """

    :param predictor:  An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param test_points:
    :param minibatch_size:
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: A PredictionResult containing the score on the test set
    """

    record = PredictionResult()

    def report_test(current_sample_number):
        current_epoch = float(current_sample_number)/dataset.training_set.n_samples
        training_cost = evaluate_predictor(predictor, test_set = dataset.training_set, evaluation_function=evaluation_function)
        test_cost = evaluate_predictor(predictor, test_set = dataset.test_set, evaluation_function=evaluation_function)
        record.add(current_epoch, [('Test', test_cost), ('Training', training_cost)])
        if report_test_scores:
            print 'Training score at epoch %s: %s' % (current_epoch, training_cost)
            print 'Test score at epoch %s: %s' % (current_epoch, test_cost)

    test_point_index = 0

    for (n_samples_seen, input_minibatch, target_minibatch) in dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = float('inf'), single_channel = True):

        if n_samples_seen >= test_points[test_point_index]:
            report_test(n_samples_seen)
            test_point_index +=1
            if test_point_index == len(test_points):
                break

        predictor.train(input_minibatch, target_minibatch)

    return record


def assess_incremental_predictor(predictor, dataset, evaluation_function, sampling_points, accumulation_function='mean', report_test_scores=True):
    """

    :param predictor: An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param sampling_points:
    :param accumulation_function:
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: A PredictionResult containing the score on the test set
    """

    assert sampling_points.dtype == 'int' and np.all(sampling_points[:-1] <= sampling_points[1:])
    record = PredictionResult()
    accumulators = {
        'mean': lambda: RunningAverage(),
        'latest': lambda: lambda x: x
        }
    training_accumulator = accumulators[accumulation_function]()
    test_accumulator = accumulators[accumulation_function]()

    pi = ProgressIndicator(sampling_points[-1], update_every = (2, 'seconds'), post_info_callback=lambda: scores)
    for epoch in xrange(sampling_points[-1]):
        if epoch in sampling_points:
            accumulated_training_output = training_accumulator(predictor.predict(dataset.training_set.input))
            training_score = evaluation_function(actual = accumulated_training_output, target = dataset.training_set.target)
            accumulated_test_output = test_accumulator(predictor.predict(dataset.test_set.input))
            test_score = evaluation_function(actual = accumulated_test_output, target = dataset.test_set.target)
            scores = [('Training', training_score), ('Test', test_score)]
            record.add(epoch, scores)
            if report_test_scores:
                print 'Training score at epoch %s: %s' % (epoch, training_score)
                print 'Test score at epoch %s: %s' % (epoch, test_score)
        predictor.train(dataset.training_set.input, dataset.training_set.target)
        pi()
    return record


class PredictionResult(object):

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
            assert isinstance(scores, list) and all(len(s)==2 for s in scores)

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
