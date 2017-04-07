from abc import ABCMeta, abstractproperty, abstractmethod

__author__ = 'peter'


class IParameterized(object):

    __metaclass__ = ABCMeta

    @abstractproperty
    def parameters(self):
        """ Returns the parameters of the object """

    def get_parameter_states(self):
        return [p.get_value() for p in self.parameters]

    def set_parameter_states(self, states):
        assert len(self.parameters)==len(states)
        for p, s in zip(self.parameters, states):
            p.set_value(s)


class IFreeEnergy(object):

    __metaclass__ = ABCMeta

    @abstractmethod
    def free_energy(self, symbolic_input):
        """
        Compute the free energy given the inputs.

        :param symbolic_input: An (n_samples, ...)
        :return: free_energy: A (n_samples, ) vector representing the free energy per sample
        """
