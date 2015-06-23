from plato.interfaces.helpers import create_shared_variable, get_theano_rng, get_named_activation_function
import numpy as np

__author__ = 'peter'


class DifferenceTargetLayer(object):


    def __init__(self, w, b, w_rev, b_rev, activation = 'tanh', rng = None):

        self.rng = get_theano_rng()
        self.w = create_shared_variable(w, name = 'w')
        self.b = create_shared_variable(b, name = 'b')
        self.w_rev = create_shared_variable(w_rev, name = 'w_rev')
        self.b_rev = create_shared_variable(b_rev, name = 'b_rev')
        self.activation = get_named_activation_function(activation)

    def forward(self, x):
        return self.activation(x.dot(self.w)+self.b)

    def backward(self, y):
        return y.dot(self.w_rev + self.b_rev)

    def get_training_function(self, cost_function):




    @classmethod
    def from_initializer(cls, w_initializer, n_in, n_out, **kwargs):

        return DifferenceTargetLayer(
            w = w_initializer((n_in, n_out)),
            b = np.zeros(n_out),
            w_rev = w_initializer((n_in, n_out)),
            b_rev = np.zeros(n_in),
            **kwargs
        )









class DifferenceTargetMLP(object):


    def __init__(self):



