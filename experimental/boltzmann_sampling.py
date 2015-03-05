from general.math import binary_permutations, sigm
import numpy as np
from scipy import weave

__author__ = 'peter'


def generate_boltzmann_parameters(n_dims, mag=1, power=1, rng = None):
    """
    Create a random symmetric matrix with zero-valued autoweights, e.g. for a Boltzmann Machine.
    :param n_dims: Number of dimensions
    :param mag: Magnitude
    :param power: Power to raise weights to (maintaining sign).  Higher means more extreme values.
    :param rng: A random number generator
    :returns w - a shape (n_dims, n_dims) array where w[i, i]==0 and w[i, j] == w[j, i] for all i, j
    """
    if rng is None:
        rng = np.random.RandomState()
    w = mag*rng.randn(n_dims, n_dims)
    w = 0.5*(w+w.T)
    w[np.arange(n_dims), np.arange(n_dims)] = 0
    w = np.sign(w)*np.abs(w)**power
    b = mag*rng.randn(n_dims)
    b = np.sign(b)*np.abs(b)**power
    assert np.array_equal(w, w.T)
    return w, b


def compute_exact_boltzmann_marginals(weights, biases):
    """
    Compute the exact marginal distribution for a Bolzmann machine.
    :param weights is a (n_nodes, n_nodes) symmetric matrix
    :param biases is a (n_nodes, ) vector
    """
    # Lets do this memory-intensive but fast
    n_nodes = weights.shape[0]
    assert n_nodes == len(biases) == weights.shape[1]
    assert np.allclose(weights, weights.T)
    bmat = binary_permutations(n_nodes).astype(float)  # (2**n_dims, n_dims)
    neg_energy = 0.5*np.einsum('ij,jk,ik->i', bmat, weights, bmat) + (bmat*biases).sum(axis=1)  # (2**n_dims, )

    exp_log_prob = np.exp(neg_energy)  # (2**n_dims, )  # Negative sign?
    marginal_likelihood = np.sum(exp_log_prob[:, None]*bmat, axis = 0)  # (n_dims, )
    normalizer = np.sum(exp_log_prob)  # Scalar
    marginals = marginal_likelihood/normalizer
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
    """
    Gibbs sampling implemented in C code.  Functionally identical to gibbs_sample_boltzmann_py_naive
    """
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
            for (int j=0; j<n_dims; j++)
                current += weights[n_dims*i+j]*x[j];
            x[i] = 1/(1+exp(-current)) > random_vals[index];
            records[index] = x[i];
        }
    """ if not block else """
    int n_dims = Nweights[0];
    for (int t = 0; t<Nrecords[0]; t++){
        for (int i = 0; i<Nrecords[1]; i++){
            int index = n_dims*t+i;
            float current = biases[i];
            for (int j=0; j<n_dims; j++)
                current += weights[n_dims*i+j]*x[j];
            records[index] = 1/(1+exp(-current)) > random_vals[index];
        }
        for (int i=0; i<n_dims; i++)  // Inefficient, but oh well
            x[i] = records[n_dims*t+i];
    }
    """
    weave.inline(code, ['records', 'x', 'currents', 'weights', 'biases', 'random_vals'], compiler = 'gcc')
    return records


def gibbs_sample_boltzmann_weave_smart(weights, biases, n_steps, rng, block = False):
    """
    Gibbs sampling on a Boltzmann Machine.  Uses a "smart" trick - that is, only update input currents when a unit
    changes.  Functionally identical to gibbs_sample_boltzmann_py_naive but 100x faster, and to
    gibbs_sample_boltzmann_weave_naive, but 2x faster.
    """
    if block:  # TODO: Implement smart version of block Gibbs, see if it's faster.  Importance: Extremely low.
        return gibbs_sample_boltzmann_weave_naive(weights, biases, n_steps, rng, block)
    n_dims = weights.shape[0]
    assert weights.shape[0] == len(biases) == weights.shape[1]
    records = np.empty((n_steps, n_dims), dtype = bool)
    x = 0.5 > rng.rand(n_dims)
    random_vals = rng.rand(*records.shape)
    currents = np.empty(n_dims)
    code = """
    int index = 0;
    int n_dims = Nweights[0];
    for (int t = 0; t<Nrecords[0]; t++){
        if (t%2048 == 0)  // This prevents numerical drift arising from our little optimization trick.
            for (int i = 0; i<n_dims; i++){
                currents[i] = biases[i];
                for (int j=0; j<n_dims; j++)
                    currents[i] += weights[n_dims*i+j]*x[j];
            }
        for (int i = 0; i<Nrecords[1]; i++){
            bool xi = 1/(1+exp(-currents[i])) > random_vals[index];
            if (xi != x[i]){
                if (xi)
                    for (int j=0; j<n_dims; j++)
                        currents[j] += weights[n_dims*i+j];
                else
                    for (int j=0; j<n_dims; j++)
                        currents[j] -= weights[n_dims*i+j];
                x[i] = xi;
                }
            records[index] = xi;
            index++;
        }
    }
    """
    weave.inline(code, ['records', 'x', 'biases', 'currents', 'weights', 'random_vals'], compiler = 'gcc')
    return records


def herded_sample_boltzmann_py_naive(weights, biases, n_steps, block = False):
    """
    :param weights is a (n_nodes, n_nodes) symmetric matrix
    :param biases is a (n_nodes, ) vector
    :param n_steps is the number of steps to run Gibbs sampling for

    returns records: A (n_steps, n_nodes) array indicating the state of the MCMC at each step.
    """
    n_dims = weights.shape[0]
    assert n_dims == len(biases) == weights.shape[1]
    records = np.empty((n_steps, n_dims))
    x = np.zeros(n_dims)
    h = np.zeros(x.shape)
    for t in xrange(n_steps):
        if block:
            p = sigm(x.dot(weights)+biases)
            h += p
            x = h > 0.5
            h -= x
        else:
            for i in xrange(n_dims):
                p = sigm(x.dot(weights[i, :])+biases[i])
                h[i] += p
                x[i] = h[i] > 0.5
                h[i] -= x[i]
        records[t, :] = x
    return records.copy()


def herded_sample_boltzmann_weave_smart(weights, biases, n_steps, block = False):
    """
    Here we use the same trick as in gibbs-smart to speed things up.
    Functionally identical* to herded_sample_boltzmann_py_naive

     * well, almost.  See test.
    """

    n_dims = weights.shape[0]
    assert weights.shape[0] == len(biases) == weights.shape[1]
    records = np.empty((n_steps, n_dims), dtype = bool)
    x = np.zeros(n_dims)
    currents = np.empty(x.shape)
    phi = np.zeros(x.shape)
    code = """
    int n_dims = Nweights[0];
    int index = 0;
    for (int t = 0; t<Nrecords[0]; t++){

        if (t%2048 == 0)  // This prevents numerical drift arizing from our little optimization trick.
            for (int i = 0; i<n_dims; i++){
                currents[i] = biases[i];
                for (int j=0; j<n_dims; j++)
                    currents[i] += weights[n_dims*i+j]*x[j];
            }
        for (int i = 0; i<Nrecords[1]; i++){
            float pi = 1/(1+exp(-currents[i]));
            phi[i] += pi;
            bool xi = phi[i] > 0.5;
            phi[i] -= xi;
            if (xi != x[i]){
                if (xi)
                    for (int j=0; j<n_dims; j++)
                        currents[j] += weights[n_dims*i+j];
                else
                    for (int j=0; j<n_dims; j++)
                        currents[j] -= weights[n_dims*i+j];
                x[i] = xi;
                }
            records[index] = xi;
            index++;
        }
    }
    """ if not block else """
    int n_dims = Nweights[0];
    for (int t = 0; t<Nrecords[0]; t++){
        if (t%2048 == 0)  // This prevents numerical drift arizing from our little optimization trick.
            for (int i = 0; i<n_dims; i++){
                currents[i] = biases[i];
                for (int j=0; j<n_dims; j++)
                    currents[i] += weights[n_dims*i+j]*x[j];
            }
        for (int i = 0; i<Nrecords[1]; i++){
            int index = n_dims*t+i;
            float pi = 1/(1+exp(-currents[i]));
            phi[i] += pi;
            bool xi = phi[i] > 0.5;
            phi[i] -= xi;
            records[index] = xi;
        }
        for (int i=0; i<n_dims; i++){  // Inefficient, but oh well
            bool x_new = records[n_dims*t+i];
            if (x_new != x[i]){
                if (x_new)
                    for (int j=0; j<n_dims; j++)
                        currents[j] += weights[n_dims*i+j];
                else
                    for (int j=0; j<n_dims; j++)
                        currents[j] -= weights[n_dims*i+j];
                x[i] = x_new;
            }
        }
    }
    """  # I don't care if it's ugly as long as it's fast.

    weave.inline(code, ['records', 'x', 'currents', 'biases', 'weights', 'phi'], compiler = 'gcc')
    return records

gibbs_sample_boltzmann = gibbs_sample_boltzmann_weave_smart
herded_sample_boltzmann = herded_sample_boltzmann_weave_smart
