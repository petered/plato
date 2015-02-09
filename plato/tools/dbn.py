from collections import namedtuple
import numpy as np
from plato.interfaces.decorators import symbolic_updater, symbolic_stateless, symbolic_standard
from plato.tools.optimizers import SimpleGradientDescent
import theano
__author__ = 'peter'

rbm = namedtuple('RBM', ['propup', 'propdown', 'get_training_fcn', 'get_free_sampling_fcn'])

def simple_rbm(visible_layer, bridge, hidden_layer):

    back_bridge = bridge.get_back_bridge()

    @symbolic_stateless
    def propup(x):
        return hidden_layer(bridge(x))

    @symbolic_stateless
    def propdown(h):
        return visible_layer(back_bridge(h))

    def get_training_fcn(n_gibbs=1, persistent = False, optimizer = SimpleGradientDescent(eta = 0.01)):

        @symbolic_updater
        def train(visible_data):

            wake_hidden = propup(visible_data)

            persistent_state = sleep_hidden = theano.shared(np.zeros(wake_hidden.tag.test_value.shape)) \
                if persistent else wake_hidden

            for _ in xrange(n_gibbs):
                sleep_visible = propdown(sleep_hidden)
                sleep_hidden = propup(sleep_visible)

            wake_energy = visible_layer.free_energy(visible_data) + bridge.free_energy(visible_data)
            sleep_energy = visible_layer.free_energy(sleep_visible) + bridge.free_energy(sleep_visible)
            cost = wake_energy - sleep_energy

            params = list(set(bridge.params + back_bridge.params))
            new_params = optimizer(cost = cost, parameters = params)

            updates = [(p, new_p) for p, new_p in zip(params, new_params)]

            if persistent:
                updates.append((persistent_state, sleep_hidden))

            return updates

        return train

    def get_free_sampling_fcn(visible_state = None, hidden_state = None, n_steps = 1):


        persistent_state = theano.shared(visible_state if hidden_state is None else hidden_state, dype = theano.config.floatX)

        if visible_state is None:  # Start from hidden
            @symbolic_standard()
            def free_sample():
                hidden = persistent_state
                for _ in xrange(n_steps):
                    visible = propdown(hidden)
                    hidden = propup(visible)
                return (visible, hidden), [(persistent_state, hidden)]
        else:  # Start from visible
            @symbolic_standard
            def free_sample():
                visible = persistent_state
                for _ in xrange(n_steps):
                    hidden = propup(visible)
                    visible = propdown(hidden)
                return (visible, hidden), [(persistent_state, visible)]
        return free_sample


    @symbolic_stateless
    def bounce(self, visible_state = None, hidden_state = None, n_steps = 1):
        assert visible_state is None != hidden_state is None, \
            "You must specify only one of hidden_state, visible state.  Not both and not neither."
