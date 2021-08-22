import pytest
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


def generate_cov(n_trials, n_channels):
    """Generate a set of cavariances matrices for test purpose"""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(n_trials, n_channels)
    A = 2 * rs.rand(n_channels, n_channels) - 1
    A /= np.linalg.norm(A, axis=1)[:, np.newaxis]
    covmats = np.empty((n_trials, n_channels, n_channels))
    for i in range(n_trials):
        covmats[i] = A @ np.diag(diags[i]) @ A.T
    return covmats, diags, A


@pytest.fixture
def covmats():
    """Generate covariance matrices for test"""
    covmats, _, _ = generate_cov(6, 3)
    return covmats


@pytest.fixture
def many_covmats():
    """Generate covariance matrices for test"""
    covmats, _, _ = generate_cov(100, 3)
    return covmats
