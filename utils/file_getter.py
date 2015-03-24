import urllib2
from StringIO import StringIO
import gzip
import sys
import os
__author__ = 'peter'

LOCAL_DIR = \
    os.path.join(os.getenv("HOME"), 'Library', 'Application Support', 'Plato') if sys.platform == 'darwin' else \
    os.path.join(os.getenv("HOME"), '.PlatoData')


def get_file(relative_name, url = None, data_transformation = None):

    relative_folder, file_name = os.path.split(relative_name)
    local_folder = os.path.join(LOCAL_DIR, relative_folder)

    try:  # Best way to see if folder exists already - avoids race condition between processes
        os.makedirs(local_folder)
    except OSError:
        pass

    full_filename = os.path.join(local_folder, file_name)

    if not os.path.exists(full_filename):
        assert url is not None, "No local copy of '%s' was found, and you didn't provide a URL to fetch it from" % (full_filename, )

        print 'Downloading file from url: "%s"...' % (url, )
        response = urllib2.urlopen(url)
        data = response.read()
        print '...Done.'
        if data_transformation is not None:
            data = data_transformation(data)
        with open(full_filename, 'w') as f:
            f.write(data)
    return full_filename


def get_local_path(relative_path):
    return os.path.join(LOCAL_DIR, relative_path)


def unzip_gz(data):
    return gzip.GzipFile(fileobj = StringIO(data)).read()

