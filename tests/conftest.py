import pytest
import numpy as np
from functools import partial

from pyriemann.datasets import make_covariances, make_masks


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
def get_covmats(rndstate):
    def _gen_cov(n_trials, n_chan):
        return make_covariances(n_trials, n_chan, rndstate,
                                return_params=False)

    return _gen_cov


@pytest.fixture
def get_covmats_params(rndstate):
    def _gen_cov_params(n_trials, n_chan):
        return make_covariances(n_trials, n_chan, rndstate, return_params=True)

    return _gen_cov_params


@pytest.fixture
def get_labels():
    def _get_labels(n_trials, n_classes):
        return np.arange(n_classes).repeat(n_trials // n_classes)

    return _get_labels


@pytest.fixture
def get_masks(rndstate):
    def _gen_masks(n_matrices, n_channels):
        return make_masks(n_matrices, n_channels, n_channels // 2, rndstate)

    return _gen_masks


def get_distances():
    distances = [
        "riemann",
        "logeuclid",
        "euclid",
        "logdet",
        "kullback",
        "kullback_right",
        "kullback_sym",
    ]
    for dist in distances:
        yield dist


def get_means():
    means = [
        "riemann",
        "logeuclid",
        "euclid",
        "logdet",
        "identity",
        "wasserstein",
        "ale",
        "harmonic",
        "kullback_sym",
    ]
    for mean in means:
        yield mean


def get_metrics():
    metrics = [
        "riemann",
        "logeuclid",
        "euclid",
        "logdet",
        "kullback_sym",
    ]
    for met in metrics:
        yield met
