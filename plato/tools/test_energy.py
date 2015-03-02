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
    h = hp < np.random.rand(*hp.shape)

    assert np.allclose(np.einsum('ij,jk,ik->i', v, w, hp), (v[:, :, None]*w[None, :, :]*hp[:, None, :]).sum(axis=1).sum(axis=1))

    get_energy = lambda v, h: -np.einsum('ij,jk,ik', v, w, hp) - v.dot(b_vis).sum() - hp.dot(b_hid).sum()

    get_free_energy = lambda v: -np.sum(np.log(1+np.exp(v.dot(w)+b_hid))) - v.dot(b_vis).sum()

    energy = get_energy(v, h)
    free_energy = get_free_energy(v)


    # So - what to use to compute the gradient?
    # Free Energy.
    # It gives a "lower variance" estimate, because it marginalizes out something something ...


    pass
    # free_energy = -np.sum(np.log(1+np.exp(v.dot(w)+b_hid)), axis = 1) - v.dot(b_vis)

    # free energy = -log(sum_over_possible_h(exp(-Energy(x, h)))
    # In an RBM, energy factors over hidden units, so :
    # sum_over_possible_h(exp(-Energy(x, h))) = sum(




    # assert np.allclose(energy, free_energy)




if __name__ == '__main__':
    test_free_energy_vs_energy()