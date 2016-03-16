import pytest
from utils.tools.iteration import minibatch_index_generator, checkpoint_minibatch_index_generator

__author__ = 'peter'
import numpy as np


def test_minibatch_index_generator():

    n_samples = 48
    n_epochs = 1.5
    minibatch_size = 5

    data = np.arange(n_samples)

    expected_total_samples = int(len(data)*n_epochs)

    for slice_when_possible in (True, False):

        i = 0
        for ix in minibatch_index_generator(n_samples = n_samples, n_epochs=n_epochs, final_treatment='truncate',
                slice_when_possible = slice_when_possible, minibatch_size=minibatch_size):
            assert np.array_equal(data[ix], np.arange(i, min(expected_total_samples, i+minibatch_size)) % n_samples)
            i += len(data[ix])
        assert i == expected_total_samples == 72

        i = 0
        for ix in minibatch_index_generator(n_samples = n_samples, n_epochs=n_epochs, final_treatment='stop',
                slice_when_possible = slice_when_possible, minibatch_size=minibatch_size):
            assert np.array_equal(data[ix], np.arange(i, min(expected_total_samples, i+minibatch_size)) % n_samples)
            i += len(data[ix])
        assert i == int(expected_total_samples/minibatch_size) * minibatch_size == 70


def test_checkpoint_minibatch_generator():
    n_samples = 48
    data = np.arange(n_samples)
    for checkpoints in ([0, 20, 30, 63, 100], [20, 30, 63, 100]):
        for slice_when_possible in (True, False):
            iterator = checkpoint_minibatch_index_generator(n_samples=n_samples, checkpoints=checkpoints, slice_when_possible=slice_when_possible)
            assert np.array_equal(data[iterator.next()], np.arange(20))
            assert np.array_equal(data[iterator.next()], np.arange(20, 30))
            assert np.array_equal(data[iterator.next()], np.arange(30, 63) % 48)
            assert np.array_equal(data[iterator.next()], np.arange(63, 100) % 48)
            try:
                iterator.next()
            except StopIteration:
                pass
            except:
                raise Exception("Failed to stop iteration")




if __name__ == '__main__':
    test_minibatch_index_generator()
    test_checkpoint_minibatch_generator()
