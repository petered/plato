#!/usr/bin/env python
"""strip outputs from an IPython Notebook

Opens a notebook, strips its output, and writes the outputless version to the original file.

Useful mainly as a git pre-commit hook for users who don't want to track output in VCS.

This does mostly the same thing as the `Clear All Output` command in the notebook UI.
"""

import io
import sys

from IPython.nbformat import current
from IPython import nbformat

def strip_output(nb):
    """strip the outputs from a notebook object"""
    nb.metadata.pop('signature', None)
    for cell in nb.worksheets[0].cells:
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'prompt_number' in cell:
            cell['prompt_number'] = None
    return nb


if __name__ == '__main__':
    filename = sys.argv[1]
    if filename.endswith('.ipynb'):
        with io.open(filename, 'r', encoding='utf8') as f:
            # nb = nbformat.read(f, 'json')
            nb = current.read(f, 'json')
        nb = strip_output(nb)
        with io.open(filename, 'w', encoding='utf8') as f:
            nbformat.write(nb, f)
            #current.write(nb, f, 'json')
