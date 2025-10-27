from functools import partial

import numpy as np
import pytest

from pyriemann.datasets import make_masks, make_matrices


def requires_module(function, name, call=None):
    """Skip a test if package is not available (decorator)."""
    call = ("import %s" % name) if call is None else call
    reason = "Test %s skipped, requires %s." % (function.__name__, name)
    try:
        exec(call) in globals(), locals()
    except Exception as exc:
        if len(str(exc)) > 0 and str(exc) != "No module named %s" % name:
            reason += " Got exception (%s)" % (exc,)
        skip = True
    else:
        skip = False
    return pytest.mark.skipif(skip, reason=reason)(function)


requires_matplotlib = partial(requires_module, name="matplotlib")
requires_seaborn = partial(requires_module, name="seaborn")


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


def get_distances():
    distances = [
        "chol",
        "euclid",
        "harmonic",
        "kullback",
        "kullback_right",
        "kullback_sym",
        "logchol",
        "logdet",
        "logeuclid",
        "riemann",
        "wasserstein",
    ]
    for dist in distances:
        yield dist


def get_means():
    means = [
        "ale",
        "euclid",
        "harmonic",
        "kullback_sym",
        "logchol",
        "logdet",
        "logeuclid",
        "riemann",
        "wasserstein",
    ]
    for mean in means:
        yield mean


def get_metrics():
    metrics = [
        "euclid",
        "logdet",
        "logeuclid",
        "kullback_sym",
        "riemann",
    ]
    for met in metrics:
        yield met
