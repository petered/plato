from experimental.demo_binary_regression import demo_create_figure

__author__ = 'peter'


def test_all_create_all_figures():

    for figure in ['1', '2A', '2B', '2C', '2D', '3A', '3B', '4A', '4B', '4C', '4D', '5A', '5B',
            '5C', '5D', '6A', '6B', 'crohns']:
        demo_create_figure(figure, test_mode = True)


if __name__ == '__main__':
    test_all_create_all_figures()
