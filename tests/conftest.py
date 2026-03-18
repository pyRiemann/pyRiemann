import numpy as np
import pytest

from pyriemann.datasets import make_masks, make_matrices


@pytest.fixture
def rndstate():
    return np.random.RandomState(1234)


@pytest.fixture
def get_mats(rndstate):
    def _gen_mat(n_matrices, n_dim, kind):
        return make_matrices(n_matrices, n_dim, kind, rndstate,
                             return_params=False)

    return _gen_mat


@pytest.fixture
def get_mats_params(rndstate):
    def _gen_mat_params(n_matrices, n_dim, kind):
        return make_matrices(n_matrices, n_dim, kind, rndstate,
                             return_params=True, eigvecs_same=True)

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


# ---- Broadcast compatibility test helpers ----

BATCH_SHAPES = [(2, 3), (2, 3, 4)]


def _make_batch_spd(batch_shape, n_dim=3, seed=42):
    """Generate SPD matrices with shape (*batch_shape, n_dim, n_dim)."""
    rs = np.random.RandomState(seed)
    n_total = int(np.prod(batch_shape))
    flat = make_matrices(n_total, n_dim, "spd", rs, return_params=False)
    return flat.reshape(*batch_shape, n_dim, n_dim)


def _make_single_spd(n_dim=3, seed=99):
    """Generate a single SPD matrix of shape (n_dim, n_dim)."""
    rs = np.random.RandomState(seed)
    return make_matrices(1, n_dim, "spd", rs, return_params=False)[0]


def _first(batch_shape):
    """Index tuple selecting the first element across all batch dims."""
    return (0,) * len(batch_shape)


def _mean_first(batch_shape):
    """Index selecting X[:, 0, 0, ...] — first batch element, all matrices."""
    return (slice(None),) + (0,) * len(batch_shape)
