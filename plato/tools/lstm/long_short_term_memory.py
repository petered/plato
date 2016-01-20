from plato.core import add_update, symbolic_multi, symbolic_simple
from plato.interfaces.decorators import symbolic_updater
from plato.interfaces.helpers import create_shared_variable, get_theano_rng, get_named_activation_function
from plato.tools.optimization.cost import mean_xe
from plato.tools.optimization.optimizers import AdaMax
import theano
import theano.tensor as tt
from theano.ifelse import ifelse
__author__ = 'peter'


class LSTMLayer(object):

    def __init__(self, w_xi, w_xf, w_xc, w_xo, w_hi, w_hf, w_hc, w_ho, w_co, b_i, b_f, b_c, b_o,
                 hidden_layer_type = 'tanh'):
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
        self._hidden_activation = get_named_activation_function(hidden_layer_type)

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
        c_can = self._hidden_activation(x.dot(self.w_xc) + h.dot(self.w_hc) + self.b_c)
        f = tt.nnet.sigmoid(x.dot(self.w_xf) + h.dot(self.w_hf) + self.b_f)
        c_next = i * c_can + f * c
        o = tt.nnet.sigmoid(x.dot(self.w_xo) + h.dot(self.w_ho) + c_next.dot(self.w_co) + self.b_o)
        h_next = o*self._hidden_activation(c_next)
        return h_next, c_next

    @symbolic_simple
    def multi_step(self, inputs, h_init = None, c_init = None, update_states = True):
        """
        Do a chain of steps and update the internal states
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
        if update_states:
            add_update(h_init, h_sequence[-1])
            add_update(c_init, c_sequence[-1])
        return h_sequence

    @property
    def parameters(self):
        return [self.w_xi, self.w_xf, self.w_xc, self.w_xo, self.w_hi, self.w_hf, self.w_hc,
            self.w_ho, self.w_co, self.b_i, self.b_f, self.b_c, self.b_o]
    
    @classmethod
    def from_initializer(cls, n_input, n_hidden, initializer_fcn, hidden_layer_type='tanh'):
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
            hidden_layer_type = hidden_layer_type
            )


class AutoencodingLSTM(object):
    """
    An LSTM that learns to predict the next element in a sequence.
    """
    def __init__(self, n_input, n_hidden, initializer_fcn, input_layer_type = 'softmax', hidden_layer_type = 'tanh'):

        self.lstm = LSTMLayer.from_initializer(n_input=n_input, n_hidden=n_hidden, initializer_fcn=initializer_fcn,
            hidden_layer_type = hidden_layer_type)
        self.w_hz = create_shared_variable(initializer_fcn, (n_hidden, n_input))
        self.b_z = create_shared_variable(0, n_input)
        self.output_activation = get_named_activation_function(input_layer_type)

    def step(self, x, h = None, c = None):

        h_next, c_next = self.lstm.step(x, h, c)
        out = self.output_activation(h_next.dot(self.w_hz)+self.b_z)  # Deref by zero just because annoyting softmax implementation.
        return out, h_next, c_next

    def get_generation_function(self, maintain_state = True, stochastic = True, rng = None):
        """
        Return a symbolic function that generates a sequence (and updates its internal state).
        :param stochastic: True to sample a onehot-vector from the output.  False to simply reinsert the
            distribution vector.
        :param rng: A seed, numpy or theano random number generator
        :return: A symbolic function of the form:
            (outputs, updates) = generate(primer, n_steps)
        """
        h_init, c_init = self.lstm.get_initial_state()
        x_init = create_shared_variable(0, shape = self.lstm.n_inputs)
        rng = get_theano_rng(rng)

        @symbolic_multi
        def generate(primer, n_steps):
            """
            Generate a sequence of outputs, and update the internal state.

            primer: A sequence to prime on.  This will overwrite the OUTPUT at
                each time step.  Note: this means the first iteration will run
                off the last output from the previous call to generate.
            n_steps: Number of steps (after the primer) to run.
            return: A sequence of length n_steps.
            """
            n_primer_steps = primer.shape[0]
            n_total_steps = n_primer_steps+n_steps

            def do_step(i, x_, h_, c_):
                y_prob, h, c = self.step(x_, h_, c_)
                y_candidate = ifelse(int(stochastic), rng.multinomial(n=1, pvals=y_prob[None, :])[0].astype(theano.config.floatX), y_prob)
                y = ifelse(i < n_primer_steps, primer[i], y_candidate)  # Note: If you get error here, you just need to prime with something on first call.
                return y, h, c
            (x_gen, h_gen, c_gen), updates = theano.scan(
                do_step,
                sequences = [tt.arange(n_total_steps)],
                outputs_info = [x_init, h_init, c_init],
                )

            for var, val in updates.items():
                add_update(var, val)

            if maintain_state:
                updates += [(x_init, x_gen[-1]), (h_init, h_gen[-1]), (c_init, c_gen[-1])]
            return x_gen[n_primer_steps:],

        return generate

    def get_training_function(self, cost_func = mean_xe, optimizer = AdaMax(alpha = 1e-3), update_states = True):
        """
        Get the symbolic function that will be used to train the AutoEncodingLSTM.
        :param cost_func: Function that takes actual outputs, target outputs and returns a cost.
        :param optimizer: Optimizer: takes cost, parameters, returns updates.
        :param update_states: If true, the hidden state is maintained between calls to the training
            function.  This makes sense if your data is coming in sequentially.
        :return:
        """
        @symbolic_updater
        def training_fcn(inputs):
            hidden_reps = self.lstm.multi_step(inputs, update_states = update_states)
            outputs = self.output_activation(hidden_reps.dot(self.w_hz)+self.b_z)
            cost = cost_func(actual = outputs[:-1], target = inputs[1:])
            optimizer(cost = cost, parameters = self.parameters)
        return training_fcn

    @property
    def parameters(self):
        return self.lstm.parameters + [self.w_hz, self.b_z]
