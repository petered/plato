__author__ = 'peter'
import numpy as np


def desc(obj):
    """
    Consise - print
    """
    if isinstance(obj, np.ndarray):
        string = '<%s with shape=%s, dtype=%s at %s%s>' % (obj.__class__.__name__, obj.shape, obj.dtype, hex(id(obj)),
            ', value = '+str(obj) if obj.size<8 else ''
            )
    else:
        string = str(obj)
    return string
