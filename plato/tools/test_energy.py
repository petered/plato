import numpy as np
__author__ = 'peter'



def test_free_energy_vs_energy():

    sigm = lambda x: 1/(1+np.exp(-x))

    n_visible = 20
    n_hidden = 30
    n_samples = 10

    w = np.random.randn(n_visible, n_hidden)
    b_vis = np.random.randn(n_visible)
    b_hid = np.random.randn(n_hidden)

    # Bernoilli-bernoulli
    v = np.random.rand(n_samples, n_visible) > 0.5
    hp = sigm(v.dot(w)+b_hid)

    energy = -np.einsum('ij,jk,ik->i', v, w, hp) - v.dot(b_vis) - hp.dot(b_hid)
    free_energy = -np.sum(np.log(1+np.exp(v.dot(w)+b_hid)), axis = 1) - v.dot(b_vis)

    assert np.allclose(energy, free_energy)




if __name__ == '__main__':
    test_free_energy_vs_energy()