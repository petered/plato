from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.scale import ScaleBase, register_scale
from matplotlib.ticker import NullFormatter, AutoLocator, ScalarFormatter, NullLocator
from matplotlib.transforms import Transform
import numpy as np

__author__ = 'peter'


def plot_learning_curves(learning_curves, xscale = 'sqrt', yscale = 'linear', hang = None, title = None, figure_name = None):
    """
    Plot a set of PredictionResults.  These can be obtained by running compare_predictors.
    See module test_compare_predictors for an example.

    :param learning_curves: An OrderedDict<str: LearningCurveData>
    :param xscale: {'linear', 'log', 'symlog', 'sqrt'}
    :param yscale: {'linear', 'log', 'symlog', 'sqrt'}
    :param hang: True for blocking plot.  False to keep executing.
    :param title: Title of the plot
    :return:
    """

    colours = ['b', 'r', 'g', 'm', 'c', 'k']

    plt.figure(figure_name)

    legend = []

    for (record_name, record), colour in zip(learning_curves.iteritems(), cycle(colours)):
        times, scores = record.get_results()
        if len(times)==1 and times[0] is None:
            assert all(len(s)==1 for s in scores.values())
            if 'Training' in scores:
                plt.axhline(scores['Training'], color=colour, linestyle = '--')
            if 'Test' in scores:
                plt.axhline(scores['Test'], color=colour, linestyle = '-')
        else:
            if 'Training' in scores:
                plt.plot(times+(1 if xscale == 'log' else 0), scores['Training'], '--'+colour)
            if 'Test' in scores:
                plt.plot(times+(1 if xscale == 'log' else 0), scores['Test'], '-'+colour)
        plt.gca().set_xscale(xscale)
        plt.gca().set_yscale(yscale)
        legend += ['%s-training' % record_name, '%s-test' % record_name]

    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend(legend, loc = 'best')
    if title is not None:
        plt.title(title)

    if hang is True:
        plt.ioff()
    elif hang is False:
        plt.ion()
    plt.show()


class SqrtScale(ScaleBase):
    """
    The Square-Root scale.  This is useful for displaying things like learning curves, which tend to have a lot of
    action early on and then flatten out later.  Unlike log-scale, it can represent zero.  The compression of larger
    values is also less dramatic than in log-scale.
    """

    name = 'sqrt'

    def __init__(self, *args, **kwargs):
        pass

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance representing the square-root transform
        """
        return SqrtTransform()

    # def limit_range_for_scale(self, vmin, vmax, minpos):
    #     """
    #     Copied from LogScale.limit_range_for_scale
    #     """
    #     return (vmin <= 0.0 and minpos or vmin,
    #             vmax <= 0.0 and minpos or vmax)

    def set_default_locators_and_formatters(self, axis):
        """
        Just took the code from LinearScale.set_default_locators_and_formatters
        """
        axis.set_major_locator(AutoLocator())
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())


class SqrtTransform(Transform):
    """
    Mostly coopied from LogTransform
    """

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self):
        Transform.__init__(self)

    def transform_non_affine(self, a):
        return np.sqrt(np.abs(a))*np.sign(a)

    def inverted(self):
        return QuadTransform()


class QuadTransform(Transform):

    input_dims = 1
    output_dims = 1
    is_separable = True
    has_inverse = True

    def __init__(self):
        Transform.__init__(self)

    def transform_non_affine(self, a):
        return a**2*np.sign(a)

    def inverted(self):
        return SqrtTransform()


register_scale(SqrtScale)
