from general.test_mode import set_test_mode
from tutorial.rbm_tutorial import demo_rbm_tutorial

__author__ = 'peter'


def test_rbm_tutorial():

    demo_rbm_tutorial()


if __name__ == '__main__':
    set_test_mode(True)
    test_rbm_tutorial()
