import sys
import os

__author__ = 'peter'


def get_local_server_dir(subdir = None):
    """
    Get the directory at the root of the venv.
    :param subdir:
    :return:
    """
    figures_dir = os.path.abspath(os.path.join(sys.executable, '..', '..', '..'))
    if subdir is not None:
        figures_dir = os.path.join(figures_dir, subdir)
    return figures_dir


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


def get_relative_link(local_path_to_file, base_dir = get_local_server_dir()):
    """
    Given a local path to a file, return a relative link path to access it from the server.
    :param full_loc: A local file path
    :param base_dir: The directory from which the server is running.
    :return: A string representing the relative link to get to that file.
    """

    assert local_path_to_file.startswith(base_dir)

    relative_loc = local_path_to_file[len(base_dir)+1:]

    return os.path.join('/files', relative_loc)
