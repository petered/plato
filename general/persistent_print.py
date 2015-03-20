import sys
from StringIO import StringIO

__author__ = 'peter'

"""
Save Print statements:

Useful in ipython notebooks where you lose output when printing to the browser.

On advice from:
http://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
"""

_ORIGINAL_STDOUT = sys.stdout


class PrintAndStoreLogger(object):
    def __init__(self):
        self.terminal = _ORIGINAL_STDOUT
        self.log = StringIO()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def read(self):
        return self.log.getvalue()

    def __getattr__(self, item):
        return getattr(self.terminal, item)


def capture_print(state = True):
    sys.stdout = PrintAndStoreLogger() if state else _ORIGINAL_STDOUT


def read_print():
    return sys.stdout.read()


def reprint():
    print read_print()
