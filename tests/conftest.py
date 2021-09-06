import pytest
from pytest import approx
import numpy as np
from functools import partial


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


def generate_cov(n_trials, n_channels, rs, return_params=False):
    """Generate a set of covariances matrices for test purpose"""
    diags = 2.0 + 0.1 * rs.randn(n_trials, n_channels)
    A = 2 * rs.rand(n_channels, n_channels) - 1
    A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
    covmats = np.empty((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        covmats[i] = A @ np.diag(diags[i]) @ A.T
    if return_params:
        return covmats, diags, A
    else:
        return covmats


@pytest.fixture
def rndstate():
    return np.random.RandomState(1234)


@pytest.fixture
def get_covmats(rndstate):
    def _gen_cov(n_trials, n_chan):
        return generate_cov(n_trials, n_chan, rndstate, return_params=False)

    return _gen_cov


@pytest.fixture
def get_covmats_params(rndstate):
    def _gen_cov_params(n_trials, n_chan):
        return generate_cov(n_trials, n_chan, rndstate, return_params=True)

    return _gen_cov_params


@pytest.fixture
def get_labels():
    def _get_labels(n_trials, n_classes):
        return np.arange(n_classes).repeat(n_trials // n_classes)

    return _get_labels


def is_positive_semi_definite(mats):
    """Check if all matrices are positive semi-definite

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are positive semi-definite
    """
    cs = mats.shape[-1]
    return np.all(np.linalg.eigvals(mats.reshape((-1, cs, cs))) >= 0.0)


def is_positive_definite(mats):
    """Check if all matrices are positive definite

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are positive definite.
    """
    cs = mats.shape[-1]
    return np.all(np.linalg.eigvals(mats.reshape((-1, cs, cs))) > 0.0)


def is_symmetric(X):
    """Check if all matrices are symmetric.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are symmetric.
    """
    # return assert_array_almost_equal(X, np.swapaxes(X, -2, -1), 6)
    return X == approx(np.swapaxes(X, -2, -1))


@pytest.fixture
def is_spd():
    """Check if all matrices are symmetric positive-definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are symmetric positive-definite.
    """

    def _is_spd(X):
        return is_symmetric(X) and is_positive_definite(X)

    return _is_spd


@pytest.fixture
def is_spsd():
    """Check if all matrices are symmetric positive semi-definite.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        The set of square matrices, at least 2D ndarray.

    Returns
    -------
    ret : boolean
        True if all matrices are symmetric positive semi-definite.
    """

    def _is_spsd(X):
        return is_symmetric(X) and is_positive_semi_definite(X)

    return _is_spsd


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
