from argmaxlab.spiking_experiments.spike_sampling import get_rng
from general.mymath import sigm, binary_permutations, cummean
import numpy as np
from sklearn.svm.classes import LinearSVC
from utils.benchmarks.train_and_test import percent_correct

__author__ = 'peter'


def logsumexp(x, axis = None):
    """
    A more numerically stable version of
    np.log(np.sum(np.exp(x))
    """
    alpha = np.max(x, axis=axis, keepdims=True) - np.log(2**63-1)
    return np.sum(alpha, axis) + np.log(np.sum(np.exp(x-alpha), axis = axis))


def logmeanexp(x, axis):
    """
    A more numerically stable version of:
    np.log(np.mean(np.exp(x), axis))
    """
    return logsumexp(x, axis=axis)-np.log(x.shape[axis])


def logcummeanexp(x, axis):
    """
    A more numerically stable version of:
    np.log(cummean(np.exp(x), axis))
    # TODO: Actually make this numerically stable.
    """
    return np.log(cummean(np.exp(x), axis))


def logdiffexp(x, axis = None):
    """
    A more numerically stable version of
    np.log(np.diff(np.exp(x), axis = axis)
    """
    alpha = np.max(x, axis=axis, keepdims=True) - np.log(2**60-1)/2
    return alpha + np.log(np.diff(np.exp(x-alpha), axis = axis))


def estimate_log_z(w, b_h, b_v, annealing_ratios, n_runs = 10, rng = None):
    """
    Use Annealed importance sampling
    http://www.iro.umontreal.ca/~lisa/pointeurs/breuleux+bengio_nc2011.pdf
    To estimate the probability of the test data given the RBM parameters.

    This code is a Pythonified version of Russ Salakhutdinov's Matlab code:
    http://www.utstat.toronto.edu/~rsalakhu/code_AIS/RBM_AIS.m

    NOTE: THIS CODE DOES NOT SEEM TO BE PRODUCING GOOD RESULTS (They don't match with exact numbers.  Not sure why!)
    Better option: Use the rbm_ais method from pylearn2 (from pylearn2.rbm_tools import rbm_ais)

    :param w: Weights (n_visible, n_hidden)
    :param b_h: Hidden biases (n_hidden)
    :param b_v: Visible biases (n_visible)
    :param annealing_ratios: A monotonically increasing vector from 0 to 1
    :param n_runs: Number of annealing chains to use.
    :param rng: Random Number generator
    :return:
    """
    assert annealing_ratios[0]==0 and annealing_ratios[-1]==1 and np.all(np.diff(annealing_ratios)>0)
    rng = get_rng(rng)
    n_visible, n_hidden = w.shape
    visbiases_base = np.zeros_like(b_v)
    neg_data = rng.rand(n_runs, n_visible) < sigm(visbiases_base)  # Collect
    logww = - neg_data.dot(visbiases_base) - n_hidden*np.log(2)
    w_h = neg_data.dot(w)+b_h
    bv_base = neg_data.dot(visbiases_base)
    bee_vee = bv_base
    for t, r in enumerate(annealing_ratios):
        exp_wh = np.exp(r*w_h)
        logww += (1-r)*bv_base + r*bee_vee + np.sum(np.log(1+exp_wh), axis =1)
        wake_hid_probs = exp_wh/(1+exp_wh)
        wake_hid_states = wake_hid_probs > rng.rand(*wake_hid_probs.shape)
        neg_vis_probs = sigm((1-r)*visbiases_base + r*(wake_hid_states.dot(w.T)+b_v))
        neg_vis_states = neg_vis_probs > rng.rand(*neg_vis_probs.shape)

        w_h = neg_vis_states.dot(w)+b_h
        bv_base = neg_vis_states.dot(visbiases_base)
        bee_vee = neg_vis_states.dot(b_v)

        exp_wh = np.exp(r*w_h)
        logww -= (1-r)*bv_base + r*bee_vee + np.sum(np.log(1+exp_wh), axis = 1)

    exp_wh = np.exp(w_h)
    logww += neg_data.dot(b_v) + np.sum(np.log(1+exp_wh), axis = 1)

    np.mean(logww)
    r_ais = logsumexp(logww) - np.log(n_runs)
    log_z_base = np.sum(np.log(1+np.exp(visbiases_base))) + n_hidden*np.log(2)
    log_z_est = r_ais + log_z_base
    aa = np.mean(logww)
    logstd_AIS = np.log(np.std(np.exp(logww-aa))) + aa - np.log(n_runs)/2
    logZZ_est_up = logsumexp([np.log(3)+logstd_AIS, r_ais], axis = 0) + log_z_base
    logZZ_est_down = logdiffexp([(np.log(3)+logstd_AIS), r_ais], axis = 0) + log_z_base
    return log_z_est, (logZZ_est_up, logZZ_est_down)


def compute_exact_log_z(w, b_h, b_v):
    """
    Compute the exact partition of an RBM.  Taken from:
    http://www.utstat.toronto.edu/~rsalakhu/code_AIS/calculate_true_partition.m
    Computation scales with 2**n_hidden, so don't use this with over ~25 hidden units!

    :param w: Weights (n_visible, n_hidden)
    :param b_h: Hidden biases (n_hidden)
    :param b_v: Visible biases (n_visible)
    :return: A scalar indicating the exact Partition.
    """
    n_visible, n_hidden = w.shape

    assert n_hidden < 26, 'Too big!'
    all_hidden = binary_permutations(n_hidden)  # 2**n_hidden * n_hidden
    log_prob_vv = all_hidden.dot(b_h) + np.sum(np.log(1+np.exp(all_hidden.dot(w.T)+b_v)), axis = 1)  #
    log_z = logsumexp(log_prob_vv, axis = 0)
    return log_z


def log_prob_data(w, b_h, b_v, log_z, data):
    """
    Given RBM parameters and the partition (log_z), compute the log-probability of the data.
    :param w: Weights (n_visible, n_hidden)
    :param b_h: Hidden biases (n_hidden)
    :param b_v: Visible biases (n_visible)
    :param log_z: Partition (a scalar) - Note that this should be a function of (w, b_h, b_v), but since this function
        takes a very long time to compute, it's provided as a separate argument, so that you can do things like approximate
        it.  If this value is very wrong, the result of this function will also be very wrong.
    :param data: An (n_samples, n_visible) array of data
    :return: The log-probability of the data given the model.
    """
    log_likelihood = data.dot(b_v) + np.sum(np.log(1+np.exp(b_h+data.dot(w))), axis = 1)
    log_prob_of_data = np.sum(log_likelihood)/data.shape[0] - log_z
    return log_prob_of_data


def get_svm_score(w, b_h, dataset):
    """
    Given a trained RBM, get the classification score of a linear SVM trained on the hidden Representation
    :param w: Weights
    :param b_h: Hidden biases
    :param dataset: A Dataset object
    :return: A scalar score
    """
    proj_training_data = sigm(dataset.training_set.input.dot(w)+b_h)
    classifier = LinearSVC()
    classifier.fit(proj_training_data, dataset.training_set.target)
    proj_test_data = sigm(dataset.test_set.input.dot(w)+b_h)
    predicted_labels = classifier.predict(proj_test_data)
    score = percent_correct(dataset.test_set.target, predicted_labels)
    return score


def count_equal_elements(model_samples, test_samples, axis = -1):
    """
    :param model_samples: (..., n_model_samples, n_dims) data
    :param test_samples: (n_test_samples, n_dims) test data
    :return: A (..., n_test_samples, n_model_samples) array containing the counts of equal elements.
    """
    assert model_samples.shape[axis] == test_samples.shape[axis]
    n_equal_elements = np.sum(test_samples[..., :, None, :] == model_samples[..., None, :, :], axis = axis)
    return n_equal_elements, model_samples.shape[axis] - n_equal_elements


def lop_p_given_n_equal(n_equal_elements, n_unequal_elements, beta):
    """
    Given the number of elements that are equal/unqeual in each pairwise binary vector comparison, return the average
        log-probability.
    :param n_equal_elements: A (..., n_test_samples, n_model_samples) array
    :param n_unequal_elements: Same shape as above, counting unequal elements.
    :param beta: A scalar in [0.5, 1)
    :return: A (n_particles, n_model_samples) vector indicating the avarage log probability of the x data given the density
        function defined by the y data and beta, where lop_p[i, j] is the average log probability of the samples up to the
        j'th sampling step from the i'th chain.
    """
    probability_per_pair = np.log(beta) * n_equal_elements + np.log(1-beta) * n_unequal_elements  # (n_particles, n_test_samples, n_model_samples)
    return np.mean(logcummeanexp(probability_per_pair, axis = -1), axis = -2)


def log_p_data(x, y, beta):
    """
    Given some collection of samples x, and some collection y, and a number beta defining the smoothness of your
    density function, return the average log-probability (per sample) of data y given the samples x.
    :param x: (n_chains, n_data_samples, n_dims) data
    :param y: (n_chains, n_test_sammples, n_dims) test data
    :param beta: The beta parameter (in 0.5...1)
    :return: A (n_particles, n_data_samples) vector indicating the avarage log probability of the x data given the density
        function defined by the y data and beta, where lop_p[i, j] is the average log probability of the samples up to the
        j'th sampling step from the i'th chain.
    """
    assert 0<=beta<1
    n_equal_elements, n_unequal_elements = count_equal_elements(x, y, axis = -1)
    return lop_p_given_n_equal(n_equal_elements, n_unequal_elements, beta)


def select_beta(model_samples, n_beta, axis=-1, max_beta_selection_samples = None):
    """
    Given samples from a model, select an optimal beta by maximizing the probability of half the samples given the other
    half.
    :param model_samples: A (n_chains, n_data_samples, n_dims) of model samples.
    :param n_beta: Number of betas (in range [0.5, 1) to try)
    :param axis: The dimension axis
    :return: A scalar "beta" representing the optimal smoothing coefficient.
    """
    beta_choices = np.linspace(0.5, 1-.5/n_beta, n_beta)
    chains = np.arange(len(model_samples))
    # Pair the samples of different chains.  Evaluate beta by maximizing samples from one chain given the other.

    if max_beta_selection_samples is not None and model_samples.shape[-2] > max_beta_selection_samples:
        sample_ixs = np.linspace(0, model_samples.shape[-2]-1, max_beta_selection_samples).astype(int)
        model_samples = model_samples[:, sample_ixs]

    n_equal_elements, n_unequal_elements = count_equal_elements(model_samples=model_samples[chains, :], test_samples=model_samples[chains[::-1]], axis = axis)  # (n_chains, n_model_samples, n_model_samples)
    probs_per_beta = [np.sum(lop_p_given_n_equal(n_equal_elements, n_unequal_elements, b)[-1], axis = 0) for b in beta_choices]
    beta_choice = beta_choices[np.argmax(probs_per_beta)]
    if beta_choice in beta_choices[[0, -1]]:
        print "WARNING: A beta on the end was selected (beta = %s).  That's fishy" % (beta_choice, )
    return beta_choice


def indirect_sampling_likelihood(model_samples, test_samples, n_beta = 100, max_beta_selection_samples=None):
    """
    The Indirect Sampling Likelihood is an estimate of the probability of the test samples given a density function
    defined by the model samples.  The motivation and approach is outlined in:
    http://www.researchgate.net/profile/Y_Bengio/publication/228849856_Unlearning_for_better_mixing/links/0f3175320aaee0819c000000.pdf

    :param model_samples: An (n_chains, n_model_samples, n_dims) array of samples from the model
    :param test_samples: An (n_test_samples, n_dims) array of samples from the test set
    :param n_beta: Number of betas (see paper) to try.  Beta defined the smoothness of the density function.
    :return: A (n_chains, n_samples) array log_p.  log_p[i,j] is the average log-probability of the test-samples given
        the density function derived from the first j model samples from chain i
        
    TODO: Test me!
    """

    assert model_samples.ndim == 3
    assert test_samples.ndim == 2
    assert model_samples.shape[-1] == test_samples.shape[-1]
    beta_choice = select_beta(model_samples = model_samples, n_beta = n_beta, axis = -1, max_beta_selection_samples=max_beta_selection_samples)
    return log_p_data(model_samples, test_samples, beta = beta_choice), beta_choice
