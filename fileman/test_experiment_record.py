import pickle
from fileman.experiment_record import ExperimentRecord, start_experiment
import numpy as np
import matplotlib.pyplot as plt


__author__ = 'peter'


def _run_experiment():

    print 'aaa'
    plt.figure('sensible defaults')
    dat = np.random.randn(4, 5)
    plt.subplot(211)
    plt.imshow(dat)
    plt.subplot(212)
    plt.imshow(dat, interpolation = 'nearest', cmap = 'gray')
    plt.show()
    print 'bbb'
    plt.plot(np.random.randn(10))
    plt.show()


def test_experiment_with():

    with ExperimentRecord(filename = 'test_exp') as exp_1:
        _run_experiment()

    assert exp_1.get_logs() == 'aaa\nbbb\n'
    figs = exp_1.show_figures()
    assert len(exp_1.get_figure_locs()) == 2

    # Now assert that you can load an experiment from file and again display the figures.
    exp_file = exp_1.get_file_path()
    with open(exp_file) as f:
        exp_1_copy = pickle.load(f)

    assert exp_1_copy.get_logs() == 'aaa\nbbb\n'
    exp_1_copy.show_figures()
    assert len(exp_1.get_figure_locs()) == 2


def test_experiment_launch():

    exp = start_experiment()
    _run_experiment()
    exp.end_and_show()
    assert len(exp.get_figure_locs()) == 2


if __name__ == '__main__':
    test_experiment_with()
    test_experiment_launch()
