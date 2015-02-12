__author__ = 'peter'
import os

LOCAL_DIR = os.path.join(os.getenv("HOME"), 'Library', 'Application Support', 'Plato')


def get_file(local_name, url = None):

    relative_folder, file_name = os.path.split(local_name)
    local_folder = os.path.join(LOCAL_DIR, relative_folder)

    try:  # Best way to see if folder exists already - avoids race condition
        os.makedirs(local_folder)
    except OSError:
        pass

    full_filename = os.path.join(local_folder, file_name)

    if os.path.exists(full_filename):
        return full_filename
    else:
        raise NotImplementedError('File "%s" not found locally and downloading from URL not implemented yet.  Implement it!' % (full_filename, ))
