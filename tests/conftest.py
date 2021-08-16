import pytest
import numpy as np


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
