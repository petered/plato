from plato.interfaces.decorators import symbolic_updater, symbolic_stateless
from plato.tools.basic import softmax
from plato.tools.networks import MultiLayerPerceptron
from plato.tools.online_prediction.online_predictors import IOnlinePredictor
from plato.tools.sampling import sample_categorical
import theano
from theano.tensor.shared_randomstreams import RandomStreams
import theano.tensor as tt
import numpy as np


__author__ = 'peter'


class GibbsSamplingMLP(IOnlinePredictor):

    def __init__(self, layer_sizes, input_size, hidden_activation = 'sig', output_activation = 'sig', w_prior = None,
            possible_ws = (0, 1), frac_to_update = 1, random_seed = None):

        assert output_activation in ('sig', 'softmax')
        if w_prior is None:
            self._w_prior = theano.shared(np.ones(len(possible_ws))/len(possible_ws))  # TODO: INCLUDE!
        self._forward_network = MultiLayerPerceptron(layer_sizes=layer_sizes, input_size=input_size, hidden_activation=hidden_activation,
                output_activation=output_activation, w_init = lambda n_in, n_out: np.zeros((n_in, n_out)))
        self._possible_ws = np.array(possible_ws).astype(theano.config.floatX)

        self._rng = RandomStreams(random_seed)
        self._frac_to_update = frac_to_update

    @symbolic_stateless
    def predict(self, input_data):
        p_y_given_xw = self._forward_network(input_data)
        return p_y_given_xw

    @symbolic_updater
    def train(self, input_data, target_data):
        """
        :param input_data: (n_samples, n_input_dims) input data
        :param target_data: (n_samples, n_output_dims) output data
        """
        network_output = self._forward_network(input_data)
        log_p_y_given_xw = tt.log(network_output[tt.arange(input_data.shape[0]), target_data]).sum(axis = 0)
        updates = [(w, self._update_w(log_p_y_given_xw, w)) for w in self._forward_network.parameters]

        return updates

    def _update_w(self, log_p_y_given_xw, param):
        p_w = self._get_p_w(log_p_y_given_xw, param)
        sampled_p_values = sample_categorical(self._rng, p_w).astype(theano.config.floatX)
        new_param = tt.switch(self._rng.uniform(param.get_value().shape)<self._frac_to_update, sampled_p_values, param)
        return new_param

    def _get_p_w(self, log_p_y_given_xw, param):
        # Return an array of shape param.shape+(n_possible_ws, ) defining the distribution over possible values of the parameter.
        expander = (slice(None), )*param.get_value().ndim + (None, )

        log_likelihoods = tt.grad(log_p_y_given_xw, param)[expander] * (self._possible_ws-param[expander])
        p_w = softmax(log_likelihoods, axis = -1) * self._w_prior

        return p_w
