from experimental.demo_deep_sampling import demo_compare_deep_samplers
from experimental.demo_binary_regression import demo_create_figure, get_figure_numbers

__author__ = 'peter'


def test_demo_compare_deep_samplers():

    demo_compare_deep_samplers(which_dataset='clusters', test_mode=True)


def test_demo_binary_regression():

    for figure in get_figure_numbers():
        print 'Testing Figure %s ...' % (figure, )
        demo_create_figure(figure, test_mode = True)
        print '... Passed.'


if __name__ == '__main__':
    test_demo_binary_regression()
    test_demo_compare_deep_samplers()
