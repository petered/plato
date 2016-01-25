from collections import namedtuple

import numpy as np
from plato.core import add_update, symbolic_multi
from plato.interfaces.decorators import symbolic_updater, symbolic_simple
from plato.tools.optimization.optimizers import SimpleGradientDescent
import theano
import theano.tensor as tt

__author__ = 'peter'

rbm = namedtuple('RBM', ['propup', 'propdown', 'get_training_fcn', 'get_free_sampling_fcn', 'vars'])


def simple_rbm(visible_layer, bridge, hidden_layer):

    @symbolic_simple
    def propup(x):
        return hidden_layer(bridge(x))

    @symbolic_simple
    def propdown(h):
        return visible_layer(bridge.reverse(h))

    def get_training_fcn(n_gibbs=1, persistent = False, optimizer = SimpleGradientDescent(eta = 0.01)):

        @symbolic_updater
        def train(wake_visible):

            wake_hidden = propup(wake_visible)

            persistent_state = sleep_hidden = theano.shared(np.zeros(wake_hidden.tag.test_value.shape, dtype = theano.config.floatX),
                name = 'persistend_hidden_state') if persistent else wake_hidden

            for _ in xrange(n_gibbs):
                sleep_visible = propdown(sleep_hidden)
                sleep_hidden = propup(sleep_visible)

            wake_energy = bridge.free_energy(wake_visible) + hidden_layer.free_energy(bridge(wake_visible))
            sleep_energy = bridge.free_energy(sleep_visible) + hidden_layer.free_energy(bridge(sleep_visible))
            cost = tt.mean(wake_energy - sleep_energy)

            params = visible_layer.parameters+bridge.parameters+hidden_layer.parameters
            optimizer(cost = cost, parameters = params, constants = [wake_visible, sleep_visible])

            if persistent:
                add_update(persistent_state, sleep_hidden)

        return train

    def get_free_sampling_fcn(init_visible_state = None, init_hidden_state = None, n_steps = 1, return_smooth_visible = False):

        assert (init_visible_state is None) != (init_hidden_state is None), \
            "You must specify only one of hidden_state, visible state.  Not both and not neither."
        start_from = 'visible' if init_hidden_state is None else 'hidden'
        persistent_state = theano.shared((init_visible_state if init_hidden_state is None else init_hidden_state).astype(theano.config.floatX),
            name = 'persistent_%s_state' % start_from)

        @symbolic_multi
        def free_sample():
            (visible_state, hidden_state), _ = get_bounce_fcn(start_from=start_from, n_steps = n_steps, return_smooth_visible = return_smooth_visible)(persistent_state)
            add_update(persistent_state, visible_state if start_from == 'visible' else hidden_state)
            return visible_state, hidden_state
        return free_sample

    def get_bounce_fcn(start_from = 'visible', n_steps = 1, return_smooth_visible = False):
        assert start_from in ('visible', 'hidden')

        @symbolic_multi
        def bounce_from_visible(visible):
            for _ in xrange(n_steps):
                hidden = propup(visible)
                visible = propdown(hidden)
            visible = visible_layer.smooth(bridge.reverse(hidden)) if return_smooth_visible else visible
            return visible, hidden

        @symbolic_multi
        def bounce_from_hidden(hidden):
            for _ in xrange(n_steps):
                visible = propdown(hidden)
                hidden = propup(visible)
            visible = visible_layer.smooth(bridge.reverse(hidden)) if return_smooth_visible else visible
            return visible, hidden

        return bounce_from_visible if start_from == 'visible' else bounce_from_hidden

    return rbm(propup, propdown, get_training_fcn, get_free_sampling_fcn, vars = locals())
