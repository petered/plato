from utils.benchmarks.train_and_test import evaluate_predictor, get_evaluation_function
from collections import OrderedDict
from utils.tools.mymath import sqrtspace
import numpy as np
from utils.tools.processors import RunningAverage
from utils.tools.progress_indicator import ProgressIndicator


def compare_predictors(dataset, online_predictor_constructors = {}, offline_predictor_constructors = {},
        incremental_predictor_constructors = {}, minibatch_size = 1, test_points = sqrtspace(0, 1, 10),
        evaluation_function = 'mse', report_test_scores = True, on_construction_callback = None):
    """
    Compare a set of predictors by running them on a dataset, and return the learning curves for each predictor.

    :param dataset: A DataSet object
    :param online_predictor_constructors: A dict<str:function> of online predictors.  An online predictor is
        sequentially fed minibatches of data and updates its parameters with each minibatch.
    :param offline_predictor_constructors: A dict<str:function> of offline predictors.  An offline predictor
        trains just once on the full training data, and then makes a prediction on the test data.  Unlike
        Online, Incremental predictors, an Offline predictor has no initial state, so it doesn't make sense
        to ask it to predict before any training has been done.
    :param incremental_predictor_constructors: A dict<str:function> of incremental predictors.  An incremental
        predictor gets the whole dataset in each pass, and updates its parameters each time.
    :param minibatch_size: Size of the minibatches to use for online predictors.  Can be:
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
    :param on_construction_callback: A function of the form callback(predictor) that is called when a predictor
        is constructed.  This may be useful for debugging.
    :return: An OrderedDict<LearningCurveData>
    """

    all_keys = online_predictor_constructors.keys()+offline_predictor_constructors.keys()+incremental_predictor_constructors.keys()
    assert len(all_keys) > 0, 'You have to give at least one predictor.'
    assert len(all_keys) == len(np.unique(all_keys)), "You have multiple predictors using the same names. Change that."
    type_constructor_dict = OrderedDict(
        [(k, ('offline', offline_predictor_constructors[k])) for k in sorted(offline_predictor_constructors.keys())] +
        [(k, ('online', online_predictor_constructors[k])) for k in sorted(online_predictor_constructors.keys())] +
        [(k, ('incremental', incremental_predictor_constructors[k])) for k in sorted(incremental_predictor_constructors.keys())]
        )

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
    for predictor_name, (predictor_type, predictor_constructor) in type_constructor_dict.iteritems():
        predictor = predictor_constructor()
        if on_construction_callback is not None:
            on_construction_callback(predictor)
        print '%s\nRunning predictor %s\n%s' % ('='*20, predictor_name, '-'*20)
        records[predictor_name] = \
            assess_offline_predictor(
                predictor=predictor,
                dataset = dataset,
                evaluation_function = evaluation_function,
                report_test_scores = report_test_scores
                ) if predictor_type == 'offline' else \
            assess_online_predictor(
                predictor=predictor,
                dataset = dataset,
                evaluation_function = evaluation_function,
                test_points = test_points,
                minibatch_size = minibatch_size[predictor_name],
                report_test_scores = report_test_scores
                ) if predictor_type == 'online' else \
            assess_incremental_predictor(
                predictor=predictor,
                dataset = dataset,
                evaluation_function = evaluation_function,
                sampling_points = np.sort(np.unique(np.ceil(test_points/dataset.training_set.n_samples).astype(int))),
                accumulation_function='mean',
                report_test_scores = report_test_scores
                )
    print 'Done!'

    return records


def assess_offline_predictor(predictor, dataset, evaluation_function, report_test_scores=True):
    """
    Train an offline predictor and return the LearningCurveData (which will not really be a curve,
    but just a point representing the final score.

    :param predictor:  An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: A  LearningCurveData containing the score on the test set
    """
    record = LearningCurveData()
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
    Train an online predictor and return the LearningCurveData.

    :param predictor:  An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param test_points:
    :param minibatch_size:
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: A  LearningCurveData containing the score on the test set
    """

    record = LearningCurveData()

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
            test_point_index += 1
            if test_point_index == len(test_points):
                break

        predictor.train(input_minibatch, target_minibatch)

    return record


def assess_incremental_predictor(predictor, dataset, evaluation_function, sampling_points, accumulation_function='mean',
        sampling_period = 1, report_test_scores=True, which_sets = 'training+test'):
    """
    Train an incremental predictor and return the LearningCurveData.

    These are a bit complicated.

    :param predictor: An IPredictor object
    :param dataset: A DataSet object
    :param evaluation_function: A function of the form: score=fcn(actual_values, target_values)
    :param sampling_points:
    :param accumulation_function:
    :param report_test_scores: Print out the test scores as they're computed (T/F)
    :return: A  LearningCurveData containing the score on the test set
    """

    assert sampling_points.dtype == 'int' and np.all(sampling_points[:-1] <= sampling_points[1:])
    assert which_sets in ('training', 'test', 'training+test')
    record = LearningCurveData()
    accumulators = {
        'mean': lambda: RunningAverage(),
        'latest': lambda: lambda x: x
        }
    training_accumulator = accumulators[accumulation_function]()
    test_accumulator = accumulators[accumulation_function]()

    pi = ProgressIndicator(sampling_points[-1], update_every = (2, 'seconds'), post_info_callback=lambda: scores)
    for epoch in xrange(sampling_points[-1]+1):
        if epoch % sampling_period == 0:
            if 'training' in which_sets:
                accumulated_training_output = training_accumulator(predictor.predict(dataset.training_set.input))
            if 'test' in which_sets:
                accumulated_test_output = test_accumulator(predictor.predict(dataset.test_set.input))

        if epoch in sampling_points:
            scores = []
            if 'training' in which_sets:
                training_score = evaluation_function(actual = accumulated_training_output, target = dataset.training_set.target)
                scores.append(('Training', training_score))
                if report_test_scores:
                    print 'Training score at epoch %s: %s' % (epoch, training_score)
            if 'test' in which_sets:
                test_score = evaluation_function(actual = accumulated_test_output, target = dataset.test_set.target)
                scores.append(('Test', test_score))
                if report_test_scores:
                    print 'Test score at epoch %s: %s' % (epoch, test_score)
            record.add(epoch, scores)
        predictor.train(dataset.training_set.input, dataset.training_set.target)
        pi()
    return record


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


# Following were meant to remove duplicated code in incremental/online/offline tests but ran into problem with incremental
#
# def _compute_scores(dataset, prediction_function, evaluation_function, which_sets):
#
#     if which_sets == 'training':
#         sets = [('Training', dataset.training_set)]
#     elif which_sets == 'test':
#         sets = [('Test', dataset.test_set)]
#     elif which_sets == 'training+test':
#         sets = [('Training', dataset.training_set), ('Test', dataset.test_set)]
#
#     scores = []
#     for set_name, data_collection in sets:
#         output = prediction_function(data_collection.input)
#         score = evaluation_function(actual = output, target = data_collection.target)
#         scores.append[(set_name, score)]
#
#     return scores
#
#
# def _report_scores(scores, epoch):
#     for set_name, score in scores:
#         print '%s score at epoch %s: %s' % (set_name, epoch, score)
