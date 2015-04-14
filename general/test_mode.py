import sys
__author__ = 'peter'

"""
A global shared "test_mode".  Allows all functions to modify their behaviour
in "test_mode".  Test mode should be set whenever the variable is set or when
things are beign run from pytest.

We do this to replace the previous solution of passing around a "test_mode" argumet
everywhere.

This is used in conjunction with conftest.py, which adds the _called_from_test flag.
"""

_TEST_MODE = False


def is_test_mode():
    return _TEST_MODE or (hasattr(sys, '_called_from_test') and sys._called_from_test)


def set_test_mode(state):
    global _TEST_MODE
    _TEST_MODE = state
