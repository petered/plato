from artemis.general.should_be_builtins import check
from plato.core import create_shared_variable, create_shared_variable_from_zeros, add_update, get_latest_update
from plato.interfaces.helpers import get_named_activation_function
from plato.tools.optimization.cost import get_named_cost_function


class ADMMNet(object):

    def __init__(self, ws, bs = None, hidden_activations='relu', output_activation='linear', loss='softmax-xe', beta = 1., gamma = 1.):
        self.ws = [create_shared_variable(w) for w in ws]
        self.bs = [create_shared_variable(b) for b in bs] if bs is not None else [create_shared_variable_from_zeros(w.shape[1]) for w in ws]
        self.h = get_named_activation_function(hidden_activations)
        self.out_act = get_named_activation_function(output_activation)
        self.final_loss = get_named_cost_function(loss)
        self.beta = [beta]*len(ws) if not isinstance(beta, (list, tuple)) else check(beta, len(beta)==len(ws))
        self.gamma = [gamma]*len(ws) if not isinstance(gamma, (list, tuple)) else check(gamma, len(gamma)==len(ws))

    @property
    def n_layers(self):
        return len(self.ws)

    def train(self, x, y):

        ass =

        zs =

        for l in xrange(1, self.n_layers+1):
            a_p = get_latest_update(ass[l-1])
            add_update(self.wl, a_p.T.dot(zs[l]))

            add_update(ass[l], )



            a_p = get_latest_update(ass[l-1])




