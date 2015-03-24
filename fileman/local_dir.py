import sys
import os

__author__ = 'peter'


"""
For dealing with files in a uniform way between machines, we have a local directory for data.

The idea is that be able to put things in the code like:

mnist = pkl.read(open('data/mnist.pkl'))

Where the path is referenced relative to the data directory on that machine.
"""

LOCAL_DIR = \
    os.path.join(os.getenv("HOME"), 'Library', 'Application Support', 'Plato') if sys.platform == 'darwin' else \
    os.path.join(os.getenv("HOME"), '.PlatoData')


def get_local_path(relative_path = ''):
    return os.path.join(LOCAL_DIR, relative_path)


def get_relative_path(local_path):
    assert local_path.startswith(LOCAL_DIR), '"%s" is not contained within the data directory "%s"' % (local_path, LOCAL_DIR)
    relative_loc = local_path[len(LOCAL_DIR)+1:]
    return relative_loc


def make_file_dir(full_file_path):
    """
    Make the directory containing the file in the given path, if it doesn't already exist
    :param full_file_path:
    :returns: The directory.
    """
    full_local_dir, _ = os.path.split(full_file_path)
    try:
        os.makedirs(full_local_dir)
    except OSError:
        pass
    return full_file_path

