from general.should_be_builtins import bad_value
from plato.interfaces.decorators import symbolic_updater, symbolic_standard, symbolic_stateless, SymbolicReturn
from plato.tools.optimization.optimizers import SimpleGradientDescent
import theano
import theano.tensor as tt
from utils.graph_utils import FactorGraph, InferencePath
import numpy as np


class DeepBeliefNet(object):
    """
    A Deep Belief Network.  This class contains methods for doing training and inference on a DBN.

    This class is made to be highly configurable, but not necessairily "easy-to-read" so it's not recommended as tutorial
    code.
    """

    def __init__(self, layers, bridges):
        assert all(src in layers and dest in layers for src, dest in bridges.viewkeys()), \
            'All bridges must project to and from layers'
        self._graph = FactorGraph(variables=layers, factors=bridges)
        self._layers = layers
        self._bridges = bridges

    def get_inference_function(self, input_layers, output_layers, path=None, smooth = False):
        """
        Create a synbolic function that does inference along a given path
        :param input_layers: A layer of list of layers whose activations will be provided to the funciton
        :param output_layers: A layer or list of layers whose activations will be returned from the function
        :param path: A path for inference.  From InferencePath documentation, path can be specified:
            (1) As a list of source/dest signals (with an optional third argument defining whether the pass should be "smooth".  e.g.
                [('vis', 'hid'), ('hid', 'ass'), ('ass', 'lab', True)]
                [('vis', 'hid'), (('hid', 'lab'), 'ass)), ('ass', ('hid', 'lab')), ('hid', 'vis', True)]
            (2) As a list of signals to compute in order:
                ['vis', 'hid', 'ass', 'lab']
        :param smooth: Boolean indicating whether to do smooth or stochastic inference, when not specified in
            steps of the path.
        :return: A symbolic function of standard form:
            (out_0, out_1, ...), [] = func(in_0, in_1, ..._)
            Which takes the activations of input_layers as inputs and returns the activations of output functions,
            and any state updates (currently no state updates).
        """
        input_layers = input_layers if isinstance(input_layers, (list, tuple)) else (input_layers, )
        output_layers = output_layers if isinstance(output_layers, (list, tuple)) else (output_layers, )

        execution_path = \
            self._graph.get_execution_path(InferencePath(path, default_smooth=smooth)) if path is not None else \
            self._graph.get_execution_path_from_io(input_layers, output_layers, default_smooth=smooth)

        @symbolic_standard
        def inference_fcn(*input_signals):
            initial_signal_dict = {lay: sig for lay, sig in zip(input_layers, input_signals)}
            computed_signal_dict = execution_path.execute(initial_signal_dict)
            return SymbolicReturn(outputs = [computed_signal_dict[lay] for lay in output_layers], updates = [])

        return inference_fcn

    def get_constrastive_divergence_function(self, visible_layers, hidden_layers, input_layers = None, up_path = None, n_gibbs = 1, persistent = False,
            method = 'free_energy', optimizer = SimpleGradientDescent(eta = 0.1)):
        """
        Make a symbolic function that does one step of contrastive divergence given a minibatch of input data.
        :param visible_layers: The visible layers of the RBM to be trained
        :param hidden_layers: The hidden layers of the RBM to be trained
        :param input_layers: The input layers (if not the same as the visible), whose activations will have to be passed
            up to the visible layers before training.
        :param up_path: The path from the input_layers to the hidden_layers (in the future this should be found
            automatically - now it is only computed automatically if there's a direct connection from input to visible)
        :param n_gibbs: Number of Gibbs block sampling steps to do
        :param persistent: True for pCD, false for regular
        :param optimizer: An IGradientOptimizer object.
        :return: A symbolic function of upate form:
            [(param_0, new_param_0), ...(persistent_state_0, new_persistent_state_0), ...] = func(in_0, in_1, ..._)
            That updates parameters in the specified RBM, and persistent state if persistent=True.
        """

        visible_layers = visible_layers if isinstance(visible_layers, (list, tuple)) else (visible_layers, )
        hidden_layers = hidden_layers if isinstance(hidden_layers, (list, tuple)) else (hidden_layers, )
        if input_layers is None:
            assert set(visible_layers).issubset(self._graph.get_input_variables()), "If you don't specify input layers, "\
                "the visible layers must be inputs to the graph.  But they are not.  Visible layers: %s, Input layers: %s" \
                % (visible_layers, self._graph.get_input_variables().keys())

        elif up_path is None:
            up_path = self.get_inference_function(input_layers = input_layers, output_layers = visible_layers)
        else:
            up_path = self._graph.get_execution_path(up_path)

        propup = self.get_inference_function(visible_layers, hidden_layers)
        free_energy = self.get_free_energy_function(visible_layers, hidden_layers)

        @symbolic_updater
        def cd_function(*input_signals):

            if input_layers is None:
                wake_visible = input_signals
            else:
                wake_visible, _ = up_path(*input_signals)
            wake_hidden, _ = propup(*wake_visible)

            initial_hidden =[theano.shared(np.zeros(wh.tag.test_value.shape, dtype = theano.config.floatX), name = 'persistent_hidden_state') for wh in wake_hidden] \
                if persistent else wake_hidden

            gibbs_path = [(hidden_layers, visible_layers)] + [(visible_layers, hidden_layers), (hidden_layers, visible_layers)] * (n_gibbs-1)
            sleep_visible, _ = self.get_inference_function(hidden_layers, visible_layers, gibbs_path)(*initial_hidden)
            sleep_hidden, _ = propup(*sleep_visible)

            all_params = sum([x.parameters for x in ([self._layers[i] for i in visible_layers]
                +[self._layers[i] for i in hidden_layers]+[self._bridges[i, j] for i in visible_layers for j in hidden_layers])], [])

            if method == 'free_energy':
                cost = free_energy(*wake_visible).mean() - free_energy(*sleep_visible).mean()
            elif method == 'energy':
                cost = tt.mean(wake_visible.T.dot(wake_hidden) - sleep_visible.T.dot(sleep_hidden))
            else:
                bad_value(method)

            updates = optimizer(cost = cost, parameters = all_params, constants = wake_visible+sleep_visible)

            if persistent:
                updates += [(p, s) for p, s in zip(initial_hidden, sleep_hidden)]

            return updates

        return cd_function

    def get_mean_sample_energies(self, visible_layers, hidden_layers):

        assert len(visible_layers)==1 and visible_layers[0].activation_fcn == 'bernoulli'
        assert len(hidden_layers)==2 and hidden_layers[0].activation_fcn == 'gaussian'

        vis_layer, = visible_layers
        hid_layer, = hidden_layers

        w = self._bridges[vis_layer, hid_layer]


        @symbolic_stateless
        def energy(visibles, hiddens):
            vis, = visibles
            hid, = hiddens



        n_samples = vis_samples.shape[0]
        return sum(v.T.dot(h))


    def get_free_energy_function(self, visible_layers, hidden_layers):
        """
        :param visible_layers: Visible layers of the RBM over which to compute the free energy
        :param hidden_layers: Hidden layers of the RBM over which to compute the free energy
        :return: A symbolic function of the form:
            free_energy = fcn(vis_0, vis_1, ...)
            Where each vis_x is a symbolic tensor of shape (n_samples, ...) and free_energy is a vector of length n_samples
            indicating the free energy per data point.
        """

        # TODO: Verify that computation is correct for all choices of vis/hidden layers
        # http://www.iro.umontreal.ca/~lisa/twiki/bin/view.cgi/Public/DBNEquations

        visible_layers = visible_layers if isinstance(visible_layers, (list, tuple)) else (visible_layers, )
        hidden_layers = hidden_layers if isinstance(hidden_layers, (list, tuple)) else (hidden_layers, )

        bridges = {(src, dest): b for (src, dest), b in self._bridges.iteritems() if src in visible_layers and dest in hidden_layers}

        @symbolic_stateless
        def free_energy(*visible_signals):
            """
            :param visible_signals: The inputs to the visible layer, each of shape (n_samples, n_dims)
            :return: A float vector representing the free energy of each sample.
            """
            visible_signals = {lay: sig for lay, sig in zip(visible_layers, visible_signals)}
            hidden_currents = {hid: sum([b(visible_signals[src]) for (src, dest), b in bridges.iteritems() if dest == hid]) for hid in hidden_layers}
            visible_contributions = [b.free_energy(visible_signals[src]) for (src, dest), b in bridges.iteritems()]
            hidden_contributions = [self._layers[hid].free_energy(hidden_currents[hid]) for hid in hidden_layers]
            # Note: Need to add another term for Gaussian RBMs, which have a the sigma parameter attached to the visible layer
            return sum(visible_contributions+hidden_contributions)

        return free_energy
