from abc import ABCMeta, abstractproperty

__author__ = 'peter'


class IParameterized(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def parameters(self):
        """ Returns the parameters of the object """
