from general.persistent_print import capture_print, read_print

__author__ = 'peter'


def test_persistent_print():

    capture_print()
    print 'aaa'
    print 'bbb'
    assert read_print()  == 'aaa\nbbb\n'


if __name__ == '__main__':

    test_persistent_print()
