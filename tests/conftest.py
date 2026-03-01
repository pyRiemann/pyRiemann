import numpy as np
import pytest

from pyriemann.datasets import make_masks, make_matrices


@pytest.fixture
def rndstate():
    return np.random.RandomState(1234)


@pytest.fixture
def get_mats(rndstate):
    def _gen_mat(n_matrices, n_dim, kind):
        return make_matrices(n_matrices, n_dim, kind, rndstate, return_params=False)

    return _gen_mat


@pytest.fixture
def get_mats_params(rndstate):
    def _gen_mat_params(n_matrices, n_dim, kind):
        return make_matrices(
            n_matrices, n_dim, kind, rndstate, return_params=True, eigvecs_same=True
        )

    return _gen_mat_params


@pytest.fixture
def get_weights(rndstate):
    def _gen_weight(n_matrices):
        return 1 + rndstate.rand(n_matrices)

    return _gen_weight


@pytest.fixture
def get_labels():
    def _get_labels(n_matrices, n_classes):
        if n_matrices % n_classes != 0:
            raise ValueError(
                "Number of matrices must be divisible by number of classes."
            )

        return np.arange(n_classes).repeat(n_matrices // n_classes)

    return _get_labels


@pytest.fixture
def get_masks(rndstate):
    def _gen_masks(n_matrices, n_channels):
        return make_masks(n_matrices, n_channels, n_channels // 2, rndstate)

    return _gen_masks


@pytest.fixture
def get_targets():
    def _gen_targets(n_matrices):
        return np.random.rand(n_matrices)

    return _gen_targets
