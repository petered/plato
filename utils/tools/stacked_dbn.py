from abc import abstractmethod
from plato.interfaces.decorators import symbolic_updater, symbolic_standard, symbolic_stateless
from plato.interfaces.helpers import get_theano_rng, create_shared_variable
from plato.tools.optimizers import SimpleGradientDescent
from plato.tools.tdb_plotting import tdbplot
from theano.compile.sharedvalue import SharedVariable
from theano.gof.graph import Variable
import theano.tensor as tt
import theano
import numpy as np
__author__ = 'peter'


def bernoulli_activation(inputs, rng):
    """
    Sigmoid-bernoulli activation function.
    :param inputs: An array of input pre-activations
    :param rng: A random number generator, or None if you just want to return probs.
    """
    p = tt.nnet.sigmoid(inputs)
    if rng is not None:
        return rng.binomial(p=p, size = p.shape, dtype = theano.config.floatX)
    else:
        return p


def linear_gaussian_activation(inputs, rng):
    if rng is None:
        return inputs
    else:
        return rng.normal(size = inputs.shape, avg = inputs, std = 1, dtype = theano.config.floatX)


class IGenerativeNet(object):

    @abstractmethod
    def propup(self, visible):
        pass

    @abstractmethod
    def propdown(self, hidden):
        pass

    @abstractmethod
    def get_training(self, n_gibbs=1, persistent = False, optimizer = SimpleGradientDescent(eta = 0.01)):
        pass

    @abstractmethod
    def get_sampling_fcn(self, initial_vis, n_steps):
        pass


class BaseRBM(IGenerativeNet):

    def __init__(self, w, b_vis, b_hid, rng):
        self.rng = get_theano_rng(rng)
        self.w = theano.shared(w)
        self.b_vis = theano.shared(b_vis)
        self.b_hid = theano.shared(b_hid)

    @property
    def parameters(self):
        return [self.w, self.b_vis, self.b_hid]

    def get_training_fcn(self, n_gibbs=1, persistent = False, optimizer = SimpleGradientDescent(eta = 0.01)):

        @symbolic_updater
        def train(wake_visible):

            wake_hidden = self.propup(wake_visible)
            persistent_state = sleep_hidden = theano.shared(np.zeros(wake_hidden.tag.test_value.shape, dtype = theano.config.floatX),
                name = 'persistend_hidden_state') if persistent else wake_hidden
            # sleep_hidden = wake_hidden


            for _ in xrange(n_gibbs):
                sleep_visible = self.propdown(sleep_hidden)
                sleep_hidden = self.propup(sleep_visible)

            wake_energy = self.energy(wake_visible)
            sleep_energy = self.energy(sleep_visible)
            cost = wake_energy - sleep_energy
            updates = optimizer(cost = cost, parameters = self.parameters, constants = [wake_visible, sleep_visible])

            if persistent:
                updates.append((persistent_state, sleep_hidden))

            return updates

        return train

    def get_sampling_fcn(self, initial_vis, n_steps):

        initial_vis = \
            theano.shared(initial_vis) if isinstance(initial_vis, np.ndarray) else \
            initial_vis if isinstance(initial_vis, SharedVariable) else \
            theano.shared(initial_vis.tag.test_value)

        @symbolic_standard
        def sample():
            vis = initial_vis
            for i in xrange(n_steps):
                hid = self.propup(vis)
                vis = self.propdown(hid)
            return (vis, hid), [(initial_vis, vis)]
        return sample

    @classmethod
    def from_initializer(cls, n_visible, n_hidden, w_init_fcn, rng = None):
        return cls(w = w_init_fcn((n_visible, n_hidden)), b_vis = np.zeros(n_visible), b_hid = np.zeros(n_hidden), rng = rng)


class BernoulliBernoulliRBM(BaseRBM):

    def propup(self, visible, stochastic = True):
        current = tt.dot(visible, self.w) + self.b_hid
        return bernoulli_activation(current, rng = self.rng if stochastic else None)

    def propdown(self, hidden, stochastic = True):
        current = tt.dot(hidden, self.w.T) + self.b_vis
        return bernoulli_activation(current, rng = self.rng if stochastic else None)

    def energy(self, visible):
        # return -tt.mean(tt.nnet.softplus(visible.dot(self.w)+self.b_hid).sum(axis = 1) + visible.dot(self.b_vis))

        hidden = self.propup(visible, stochastic = False)
        return -tt.mean(tt.sum(visible.dot(self.w)*hidden, axis = 1) + visible.dot(self.b_vis) + hidden.dot(self.b_hid))


class BernoulliGaussianRBM(BaseRBM):

    def propup(self, visible, stochastic = True):
        current = visible.dot(self.w) + self.b_hid

        return linear_gaussian_activation(current, rng = self.rng if stochastic else None)

    def propdown(self, hidden, stochastic = True):
        current = hidden.dot(self.w.T) + self.b_vis
        return bernoulli_activation(current, rng = self.rng if stochastic else None)

    def energy(self, visible):
        hidden = self.propup(visible, stochastic = False)
        return -tt.mean(tt.sum(visible.dot(self.w)*hidden, axis = 1) + visible.dot(self.b_vis) + tt.sum(0.5*(hidden - self.b_hid)**2, axis = 1))


class StackedDeepBeliefNet(IGenerativeNet):

    def __init__(self, rbms):
        self.rbms = rbms

    @symbolic_stateless
    def propup(self, visible, stochastic=True, to_layer = None):
        data = visible
        for rbm in self.rbms[:to_layer]:
            data = rbm.propup(data, stochastic = stochastic)
        return data

    def propdown(self, hidden, stochastic, from_layer = -1):
        data = hidden
        for rbm in self.rbms[from_layer::-1]:
            data = rbm.propdown(data)
        return data

    def get_training_fcn(self, **cd_params):

        @symbolic_updater
        def train(visible):
            top_rbm_visible = self.propup(visible, stochastic=False, to_layer=-1)
            top_rbm_training_fcn = self.rbms[-1].get_training_fcn(**cd_params)
            return top_rbm_training_fcn(top_rbm_visible)
        return train

    def get_sampling_fcn(self, initial_vis, n_steps):
        initial_vis = theano.shared(initial_vis)
        initial_top_vis = self.propup(initial_vis, to_layer=-1)
        top_sampling_fcn = self.rbms[-1].get_sampling_fcn(initial_vis= initial_top_vis, n_steps=n_steps)
        @symbolic_standard
        def sample():
            (top_sample, _), updates = top_sampling_fcn()
            bottom_sample = self.propdown(top_sample, stochastic = True, from_layer = -2)
            return (bottom_sample, ), updates
        return sample

    def stack_another(self, rbm):
        '''
        :param rbm: Return a stacked DBN with one more layer
        :return:
        '''
        return StackedDeepBeliefNet(self.rbms + [rbm])


