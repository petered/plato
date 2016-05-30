"""
Sets up a basic work environment, without any of the theano-based stuff.

Useful for doing imports in IPython - just go:

    from basic_environment import *

"""

import numpy as np
import matplotlib.pyplot as plt
from artemis.plotting.live_plotting import LiveStream, LiveCanal
from artemis.plotting.easy_plotting import ezplot
