"""
Takes all .py files in the directory, and, if they don't already have a
corresponding .ipynb file, creates them from the py files.  We do this because
we don't want to version-control .ipynb files (which can contain images and use
a special machine-readable format), but we do want to save their contents.

Solution: Start ipython notebook with the "--script" argument, which causese notebooks
to be automatically saved as .py files - these can be versioned.  Then when we run this
script, we regenerate all .pynb notebooks from the .py files.  The output of course will
not be preserved through this process.

This follows from a discussion here:
http://stackoverflow.com/questions/18734739/using-ipython-notebooks-under-version-control

Call this script with argument "-ov" to overwrite existing .ipynb files.
"""
import sys
import IPython.nbformat.v3 as nbf
import os
THIS_FILE = os.path.abspath(__file__)


def py_to_pynb(filename, skip_if_exists = True):
    """
    Convert a given py file into an ipynb file.  If skip_if_exists is True, and
    the ipynb file is already there, then don't do the conversion.
    """
    file_name_headless, ext = os.path.splitext(filename)
    assert filename.endswith('.py')
    pynb_filename = file_name_headless+'.ipynb'
    if os.path.exists(pynb_filename) and skip_if_exists:
        return

    with open(filename, 'r') as f:
        nb = nbf.read_py(f.read())

    with open(pynb_filename, 'w') as f:
        nbf.to_notebook_py(nb, )
    print 'Created Notebook file: %s' % (pynb_filename, )


def all_py_to_pynb(directory, skip_if_exists = True):
    """
    Convert all py files in the given directory to ipynb files
    """

    for filename in os.listdir(directory):
        if filename.endswith(".py"):
            full_filename = os.path.join(directory, filename)
            if full_filename != THIS_FILE:  # Haha
                py_to_pynb(full_filename, skip_if_exists = skip_if_exists)


if __name__ == '__main__':

    args = sys.argv
    overwrite = len(args)==2 and args[1]=='-ov'
    overwrite = True
    assert len(args)==1 or overwrite
    this_dir, _ = os.path.split(THIS_FILE)
    all_py_to_pynb(this_dir, skip_if_exists = not overwrite)
