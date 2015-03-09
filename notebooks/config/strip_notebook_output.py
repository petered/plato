# #!/usr/bin/env python
# """strip outputs from an IPython Notebook
#
# Opens a notebook, strips its output, and writes the outputless version to the original file.
#
# Useful mainly as a git pre-commit hook for users who don't want to track output in VCS.
#
# This does mostly the same thing as the `Clear All Output` command in the notebook UI.
# """
#
# import io
# import sys
#
# from IPython.nbformat import current
# from IPython import nbformat
#
# def strip_output(nb):
#     """strip the outputs from a notebook object"""
#     nb.metadata.pop('signature', None)
#     for cell in nb.worksheets[0].cells:
#         if 'outputs' in cell:
#             cell['outputs'] = []
#         if 'prompt_number' in cell:
#             cell['prompt_number'] = None
#     return nb
#
#
#
# if __name__ == '__main__':
#     filename = sys.argv[1]
#     if filename.endswith('.ipynb'):
#         with io.open(filename, 'r', encoding='utf8') as f:
#             # nb = nbformat.read(f, as_version=4)
#             nb = current.read(f, 'json')
#         nb = strip_output(nb)
#         with io.open(filename, 'w', encoding='utf8') as f:
#             nbformat.write(nb, f)
#             #current.write(nb, f, 'json')


#!/usr/bin/env python
"""strip outputs from an IPython Notebook

Opens a notebook, strips its output, and writes the outputless version to the original file.

Useful mainly as a git pre-commit hook for users who don't want to track output in VCS.

This does mostly the same thing as the `Clear All Output` command in the notebook UI.

Adapted from rom https://gist.github.com/minrk/6176788 to work with
git filter driver
"""
import sys

#You may need to do this for your script to work with GitX or Tower:
#sys.path.append("/Users/chris/anaconda/envs/conda/lib/python2.7/site-packages")

from IPython.nbformat import v4

def strip_output(nb):
    """strip the outputs from a notebook object"""
    for cell in nb.cells:
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'execution_count' in cell:
            cell['execution_count'] = 0
    return nb

if __name__ == '__main__':
    nb = v4.reads(sys.stdin.read())
    nb = strip_output(nb)
    sys.stdout.write(v4.writes(nb))
