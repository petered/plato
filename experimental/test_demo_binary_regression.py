from experimental.demo_binary_regression import demo_create_figure, get_figure_numbers

__author__ = 'peter'


def test_all_create_all_figures():

    for figure in get_figure_numbers():
        print 'Testing Figure %s ...' % (figure, )
        demo_create_figure(figure, test_mode = True)
        print '... Passed.'

if __name__ == '__main__':
    test_all_create_all_figures()
