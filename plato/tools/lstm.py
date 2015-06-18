from plato.interfaces.decorators import symbolic_updater, symbolic_standard
from plato.interfaces.helpers import initialize_param, create_shared_variable, get_theano_rng
from plato.tools.basic import softmax
from plato.tools.cost import negative_log_likelihood_dangerous, mean_xe
from plato.tools.linking import Chain
from plato.tools.networks import FullyConnectedBridge, Layer
from plato.tools.optimizers import AdaMax
import numpy as np
from plato.tools.tdb_plotting import tdbplot
import theano
import theano.tensor as tt
__author__ = 'peter'


class LSTMLayer(object):

    def __init__(self, w_xi, w_xf, w_xc, w_xo, w_hi, w_hf, w_hc, w_ho, w_co, b_i, b_f, b_c, b_o):
        """
        :param w_xi:
        :param w_xf:
        :param w_xc:
        :param w_xo:
        :param w_hi:
        :param w_hf:
        :param w_hc:
        :param w_ho:
        :param w_co:
        :param b_i:
        :param b_f:
        :param b_c:
        :param b_o:
        :return:
        """
        self.n_inputs, self.n_hidden = w_xi.get_value().shape

        self.w_xi = w_xi
        self.w_xf = w_xf
        self.w_xc = w_xc
        self.w_xo = w_xo
        self.w_hi = w_hi
        self.w_hf = w_hf
        self.w_hc = w_hc
        self.w_ho = w_ho
        self.w_co = w_co
        self.b_i = b_i
        self.b_f = b_f
        self.b_c = b_c
        self.b_o = b_o

    def get_initial_state(self, h_init = None, c_init = None):
        if h_init is None:
            h_init = create_shared_variable(0, shape = self.n_hidden, name = 'h')
        if c_init is None:
            c_init = create_shared_variable(0, shape = self.n_hidden, name = 'c')
        return h_init, c_init

    def step(self, x, h = None, c = None):
        """
        One step of LSTM processing.  Based on tutorial from:
        http://deeplearning.net/tutorial/lstm.html

        :param x: (n_dims, ) input
        :param h: (n_hidden, ) hidden activation from last step
        :param c: (n_hidden, ) memories from the last step
        :return:
        """

        h, c = self.get_initial_state(h, c)
        i = tt.nnet.sigmoid(x.dot(self.w_xi) + h.dot(self.w_hi) + self.b_i)
        c_can = tt.tanh(x.dot(self.w_xc) + h.dot(self.w_hc) + self.b_c)
        f = tt.nnet.sigmoid(x.dot(self.w_xf) + h.dot(self.w_hf) + self.b_f)
        c_next = i * c_can + f * c
        o = tt.nnet.sigmoid(x.dot(self.w_xo) + h.dot(self.w_ho) + c_next.dot(self.w_co) + self.b_o)
        h_next = o*tt.tanh(c_next)
        return h_next, c_next

    @symbolic_standard
    def multi_step(self, inputs, h_init = None, c_init = None):
        """
        Do a chain of steps.
        inputs is a symbolic (n_frames, ...) array
        outputs is a symbolic (n_frames, ...) array
        """
        h_init, c_init = self.get_initial_state(h_init, c_init)
        all_states, updates = theano.scan(
            self.step,
            sequences=[inputs],
            outputs_info = [h_init, c_init],
            )
        h_sequence, c_sequence = all_states
        return (h_sequence, ), [(h_init, h_sequence[-1]), (c_init, c_sequence[-1])]

    @property
    def parameters(self):
        return [self.w_xi, self.w_xf, self.w_xc, self.w_xo, self.w_hi, self.w_hf, self.w_hc,
            self.w_ho, self.w_co, self.b_i, self.b_f, self.b_c, self.b_o]
    
    @classmethod
    def from_initializer(cls, n_input, n_hidden, initializer_fcn):
        """
        :param n_input: Number of inputs
        :param n_hidden: Number of hiddens
        :param n_output: Number of outputs
        :param initializer_fcn: Function taking a shape and returning parameters.
        :return: An LSTMLayer
        """
        return LSTMLayer(
            w_xi = create_shared_variable(initializer_fcn, shape = (n_input, n_hidden)),
            w_xf = create_shared_variable(initializer_fcn, shape = (n_input, n_hidden)),
            w_xc = create_shared_variable(initializer_fcn, shape = (n_input, n_hidden)),
            w_xo = create_shared_variable(initializer_fcn, shape = (n_input, n_hidden)),
            w_hi = create_shared_variable(initializer_fcn, shape = (n_hidden, n_hidden)),
            w_hf = create_shared_variable(initializer_fcn, shape = (n_hidden, n_hidden)),
            w_hc = create_shared_variable(initializer_fcn, shape = (n_hidden, n_hidden)),
            w_ho = create_shared_variable(initializer_fcn, shape = (n_hidden, n_hidden)),
            w_co = create_shared_variable(initializer_fcn, shape = (n_hidden, n_hidden)),
            b_i = create_shared_variable(0, shape = n_hidden),
            b_f = create_shared_variable(0, shape = n_hidden),
            b_c = create_shared_variable(0, shape = n_hidden),
            b_o = create_shared_variable(0, shape = n_hidden),
            )


class AutoencodingLSTM(object):

    def __init__(self, n_input, n_hidden, initializer_fcn, input_layer_type = 'softmax'):

        self.lstm = LSTMLayer.from_initializer(n_input=n_input, n_hidden=n_hidden, initializer_fcn=initializer_fcn)
        self.w_hz = create_shared_variable(initializer_fcn, (n_hidden, n_input))
        self.b_z = create_shared_variable(0, n_input)
        self.output_activation = {
            'softmax': lambda x: softmax(x, axis = -1),
            'sigm': tt.nnet.sigmoid
            }[input_layer_type]

    def step(self, x, h = None, c = None):

        h_next, c_next = self.lstm.step(x, h, c)
        out = self.output_activation(h_next.dot(self.w_hz)+self.b_z)  # Deref by zero just because annoyting softmax implementation.
        return out, h_next, c_next

    def get_generation_function(self, maintain_state = False, primer = None, stochastic = True, seed = None):

        x = tt.zeros(self.lstm.n_inputs)
        h, c = self.lstm.get_initial_state()

        rng = get_theano_rng(seed)

        def do_step(x_, h_, c_):
            y, h, c = self.step(x_, h_, c_)
            x = rng.multinomial(n=1, pvals=y[None, :])[0].astype(theano.config.floatX)  # Weird indexing is to get around weird restriction with multinomial.
            return x, h, c

        step_fcn = do_step if stochastic else self.step

        @symbolic_standard
        def generate(n_steps):
            outputs, updates = theano.scan(
                step_fcn,
                outputs_info = [{'initial': x}, {'initial': h}, {'initial': c}],
                n_steps=n_steps
                )
            return outputs, updates
        return generate

    def get_training_function(self, cost_func = mean_xe, optimizer = AdaMax(alpha = 1e-3), update_states = False):
        """
        Get the symbolic function that will be used to train the AutoEncodingLSTM.
        :param cost_func: Function that takes actual outputs, target outputs and returns a cost.
        :param optimizer: Optimizer: takes cost, parameters, returns updates.
        :param update_states:
        :return:
        """
        @symbolic_updater
        def training_fcn(inputs):
            (hidden_reps, ), state_updates = self.lstm.multi_step(inputs)
            outputs = self.output_activation(hidden_reps.dot(self.w_hz)+self.b_z)
            cost = cost_func(actual = outputs[:-1], target = inputs[1:])
            parameter_updates = optimizer(cost = cost, parameters = self.parameters)
            if update_states:
                return parameter_updates + state_updates
            else:
                return parameter_updates
        return training_fcn

    @property
    def parameters(self):
        return self.lstm.parameters + [self.w_hz, self.b_z]
