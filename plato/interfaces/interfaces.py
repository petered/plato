from abc import ABCMeta, abstractproperty, abstractmethod

__author__ = 'peter'


class IParameterized(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def parameters(self):
        """ Returns the parameters of the object """


class IFreeEnergy(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def free_energy(self, symbolic_input):
        """
        Compute the free energy given the inputs.

        :param symbolic_input: An (n_samples, ...)
        :return: free_energy: A (n_samples, ) vector representing the free energy per sample
        """
