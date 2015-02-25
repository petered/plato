"""
Imports useful modules and functions in the plato project.

This is mainly useful in IPython notebooks, to avoid the redundancy of always specifying
the same basic imports for every notebook, and the "hidden" behaviour of importing in the
profile setup.  Instead, just put:

    from plato_environment import *

At the top of the notebook.
"""

import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
from plato.interfaces.decorators import symbolic_stateless, symbolic_standard, symbolic_updater
import plato.tools.all as pt
from plotting.live_plotting import LiveStream, LiveCanal
from plotting.easy_plotting import ezplot
