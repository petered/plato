import os
from fileman.persistent_print import capture_print, read_print, new_log_file

__author__ = 'peter'


def test_persistent_print():

    test_log_path = capture_print(to_file=True)
    print 'aaa'
    print 'bbb'
    assert read_print()  == 'aaa\nbbb\n'
    capture_print(False)

    capture_print(True)
    assert read_print() == ''
    print 'ccc'
    print 'ddd'
    assert read_print()  == 'ccc\nddd\n'

    os.remove(test_log_path)


def test_new_log_file():
    # Just a shorthand for persistent print.

    log_file_loc = new_log_file('dump/test_file')
    print 'eee'
    print 'fff'
    capture_print(False)

    with open(log_file_loc) as f:
        text = f.read()

    assert text == 'eee\nfff\n'
    os.remove(log_file_loc)


if __name__ == '__main__':

    test_persistent_print()
    test_new_log_file()