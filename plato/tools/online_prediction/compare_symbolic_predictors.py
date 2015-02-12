from collections import namedtuple
import itertools
from plato.interfaces.decorators import symbolic_stateless
import numpy as np
from matplotlib import pyplot as pp


class CompareOnlinePredictors(object):

    def __init__(self, dataset, classifier_constructors, minibatch_size, test_points, evaluation_function):
        """
        :param dataset: A DataSet object
        :param classifier_constructors: A dict<str:function> where the functions construct IClassifier objects
        :param minibatch_size: Size of the minibatches to use.  Can be:
            An int, in which case it represents the minibatch size for all classifiers.
            A dict<str: int>, in which case you can set the minibatch size per-classifier.
        :param test_points: Test points to use.  Can be:
            A list of integers - in which case the classifier is after seeing this many samples.
            A list of floats - in which case the classifier is tested after seeing this many epochs.
            The final test point determines the end of training.
        :param evaluation_function: Symbolic function used to evaluate output of classifiers
        :return:
        """

        self._dataset = dataset
        self._classifier_constructors = classifier_constructors
        if isinstance(minibatch_size, int):
            self._minibatch_sizes = {classifier_name: minibatch_size for classifier_name in classifier_constructors.keys()}
        else:
            assert classifier_constructors.viewkeys() == minibatch_size.viewkeys()
        self._minibatch_size = minibatch_size
        test_points = np.array(test_points)
        self._test_points_float = test_points.dtype == float
        if self._test_points_float:
            test_points = (test_points * self._dataset.training_set.n_samples).astype(int)

        self._test_points = test_points
        self._evaluation_function = evaluation_function

    def __call__(self):

        records = {}

        for classifier_name, cc in self._classifier_constructors.iteritems():

            print '%s\nRunning classifier %s\n%s' % ('='*20, classifier_name, '-'*20)

            minibatch_size = self._minibatch_size[classifier_name]

            record = Record([], [])
            records[classifier_name] = record

            classifier = cc()
            training_fcn = classifier.train.compile()

            @symbolic_stateless
            def symbolic_evaluation_fcn(inputs, labels):
                predictions = classifier.classify(inputs)
                score = self._evaluation_function(predictions, labels)
                return score

            evaluation_fcn = symbolic_evaluation_fcn.compile()

            def report_test(current_sample_number):
                current_epoch = float(current_sample_number)/self._dataset.training_set.n_samples
                training_cost = evaluation_fcn(self._dataset.training_set.input, self._dataset.training_set.target)
                print 'Training score at epoch %s: %s' % (current_epoch, training_cost)
                test_cost = evaluation_fcn(self._dataset.test_set.input, self._dataset.test_set.target)
                print 'Test score at epoch %s: %s' % (current_epoch, test_cost)
                record.training_score.append((current_epoch, training_cost))
                record.test_score.append((current_epoch, test_cost))

            test_points = (self._test_points * self._dataset.training_set.n_samples).astype(int)
            test_point_index = 0

            for i, (input_minibatch, label_minibatch) in enumerate(self._dataset.training_set.minibatch_iterator(minibatch_size = minibatch_size, epochs = float('inf'), single_channel = True)):

                current_sample = i * minibatch_size
                if current_sample >= self._test_points[test_point_index]:
                    report_test(current_sample)
                    test_point_index +=1
                    if test_point_index == len(test_points):
                        break

                training_fcn(input_minibatch, label_minibatch)

        print 'Done!'

        return records


Record = namedtuple('TrainingRecord', ('training_score', 'test_score'))


def plot_records(records):

    colours = ['r', 'g', 'b', 'k', 'p']

    pp.figure()

    legend = []

    for (record_name, record), colour in zip(records.iteritems(), colours):

        training_times = np.array([t for t, _ in record.training_score])
        training_scores = np.array([s for _, s in record.training_score])
        test_times = np.array([t for t, _ in record.test_score])
        test_scores = np.array([s for _, s in record.test_score])

        pp.plot(test_times, test_scores, '-'+colour)
        pp.plot(training_times, training_scores, '--'+colour)
        legend+=['%s-test' % record_name, '%s-training' % record_name]

    pp.xlabel('Epoch')
    pp.ylabel('Score')
    pp.legend(legend, loc = 'best')
    pp.show()
