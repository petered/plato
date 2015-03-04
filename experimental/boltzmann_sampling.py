from general.math import binary_permutations, sigm
import numpy as np
from scipy import weave

__author__ = 'peter'


def random_symmetric_matrix(n_dims, rng, mag=1, power=1):
    w = mag*np.random.randn(n_dims, n_dims)
    w = 0.5*(w+w.T)
    w[np.arange(n_dims), np.arange(n_dims)] = 0
    w = np.sign(w)*np.abs(w)**power
    return w


def compute_exact_boltzmann_marginals(weights, biases):
    """
    weights is a (n_nodes, n_nodes) symmetric matrix
    biases is a (n_nodes, ) vector
    """
    # Lets do this memory-intensive but fast
    n_nodes = weights.shape[0]
    assert n_nodes == len(biases) == weights.shape[1]
    assert np.allclose(weights, weights.T)
    bmat = binary_permutations(n_nodes).astype(float)  # (2**n_dims, n_dims)
    energy = 0.5*np.einsum('ij,jk,ik->i', bmat, weights, bmat) + (bmat*biases).sum(axis=1)  # (2**n_dims, )

    exp_log_prob = np.exp(energy)  # (2**n_dims, )  # Negative sign?
    q = np.sum(exp_log_prob[:, None]*bmat, axis = 0)  # (n_dims, )
    normalizer = np.sum(exp_log_prob)  # Scalar
    marginals = q/normalizer
    return marginals


def gibbs_sample_boltzmann_py_naive(weights, biases, n_steps, rng, block = False):
    """
    weights is a (n_nodes, n_nodes) symmetric matrix
    biases is a (n_nodes, ) vector
    n_steps is the number of steps to run Gibbs sampling for
    rng is a random number generator

    returns records: A (n_steps, n_nodes) array indicating the state of the MCMC at each step.
    """
    n_dims = weights.shape[0]
    assert n_dims == len(biases) == weights.shape[1]
    records = np.empty((n_steps, n_dims))
    x = 0.5 > rng.rand(n_dims)
    for t in xrange(n_steps):
        if block:
            x = sigm(x.dot(weights)+biases) > rng.rand(n_dims)
        else:
            for i in xrange(n_dims):
                x[i] = sigm(x.dot(weights[i, :])+biases[i]) > rng.rand()
        records[t, :] = x
    return records


def gibbs_sample_boltzmann_py_smart(weights, biases, n_steps, rng, block = False):
    """
    weights is a (n_nodes, n_nodes) symmetric matrix
    biases is a (n_nodes, ) vector
    n_steps is the number of steps to run Gibbs sampling for
    rng is a random number generator

    returns records: A (n_steps, n_nodes) array indicating the state of the MCMC at each step.
    """
    if block:
        return gibbs_sample_boltzmann_py_naive(weights, biases, n_steps, rng, block)
    n_dims = weights.shape[0]
    assert n_dims == len(biases) == weights.shape[1]
    records = np.empty((n_steps, n_dims))
    x = 0.5 > rng.rand(n_dims)
    input_currents = x.dot(weights)+biases  # Recomputing just to avoid drift
    for t in xrange(n_steps):
        if block:
            x = sigm(input_currents) > rng.rand(n_dims)
        else:
            for i in xrange(n_dims):
                xi = sigm(input_currents[i]) > rng.rand()
                # if xi != x[i]:
                #     input_currents+=(xi*2-1)*weights[i, :]
                #     x[i]=xi

                input_currents+=(int(xi)-x[i])*weights[i, :]
                x[i]=xi
        records[t, :] = x
    return records


def gibbs_sample_boltzmann_weave_naive(weights, biases, n_steps, rng, block = False):
    n_dims = weights.shape[0]
    assert weights.shape[0] == len(biases) == weights.shape[1]
    # assert not block, 'Not implemented for block-sampling'
    records = np.zeros((n_steps, n_dims), dtype = bool)
    x = 0.5 > rng.rand(n_dims)
    random_vals = rng.rand(*records.shape)
    currents = x.dot(weights)+biases
    code = """
    int n_dims = Nweights[0];
    for (int t = 0; t<Nrecords[0]; t++)
        for (int i = 0; i<Nrecords[1]; i++){
            int index = n_dims*t+i;
            float current = biases[i];
            int w_ptr = n_dims*i;
            for (int j=0; j<n_dims; j++)
                current += weights[w_ptr+j]*x[j];
            x[i] = 1/(1+exp(-current)) > random_vals[index];
            records[index] = x[i];
        }
    """ if not block else """
    int n_dims = Nweights[0];
    for (int t = 0; t<Nrecords[0]; t++){
        for (int i = 0; i<Nrecords[1]; i++){
            int index = n_dims*t+i;
            float current = biases[i];
            int w_ptr = n_dims*i;
            for (int j=0; j<n_dims; j++)
                current += weights[w_ptr+j]*x[j];
            records[index] = 1/(1+exp(-current)) > random_vals[index];
        }
        for (int i=0; i<n_dims; i++)  // Inefficient, but oh well
            x[i] = records[n_dims*t+i];
    }
    """
    weave.inline(code, ['records', 'x', 'currents', 'weights', 'biases', 'random_vals'], compiler = 'gcc')
    return records


def gibbs_sample_boltzmann_weave_smart(weights, biases, n_steps, rng, block = False):
    if block:  # TODO: Implement smart version of block Gibbs, see if it's faster.  Importance: Extremely low.
        return gibbs_sample_boltzmann_weave_naive(weights, biases, n_steps, rng, block)
    n_dims = weights.shape[0]
    assert weights.shape[0] == len(biases) == weights.shape[1]
    records = np.zeros((n_steps, n_dims), dtype = bool)
    x = 0.5 > rng.rand(n_dims)
    random_vals = rng.rand(*records.shape)
    currents = x.dot(weights)+biases
    code = """
    int n_dims = Nweights[0];
    for (int t = 0; t<Nrecords[0]; t++)
        for (int i = 0; i<Nrecords[1]; i++){
            int index = n_dims*t+i;
            bool xi = 1/(1+exp(-currents[i])) > random_vals[index];
            if (xi != x[i]){
                int w_ptr = n_dims*i;
                if (xi)
                    for (int j=0; j<n_dims; j++)
                        currents[j] += weights[w_ptr+j];
                else
                    for (int j=0; j<n_dims; j++)
                        currents[j] -= weights[w_ptr+j];
                x[i] = xi;
                }
        records[index] = xi;
        }
    """
    weave.inline(code, ['records', 'x', 'currents', 'weights', 'random_vals'], compiler = 'gcc')
    return records


gibbs_sample_boltzmann = gibbs_sample_boltzmann_weave_smart
