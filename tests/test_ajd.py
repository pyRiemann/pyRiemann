from numpy.testing import assert_array_almost_equal, assert_array_equal
from nose.tools import assert_raises
import numpy as np

from pyriemann.utils.ajd import _get_sample_weight, rjd, ajd_pham, uwedge


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose"""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats, diags, A


def test_get_sample_weight():
    """Test get_sample_weight"""
    Nt = 100
    covmats, diags, A = generate_cov(Nt, 3)
    w = np.random.randn(Nt)
    assert_raises(ValueError, _get_sample_weight, w, covmats) # non-positive
    assert_raises(ValueError, _get_sample_weight, w[:Nt//2], covmats)


def test_rjd():
    """Test rjd"""
    covmats, diags, A = generate_cov(100, 3)
    V, D = rjd(covmats)


def test_pham():
    """Test pham's ajd"""
    Nt = 100
    covmats, diags, A = generate_cov(Nt, 3)
    V, D = ajd_pham(covmats)

    w = 5 * np.ones(Nt)
    Vw, Dw = ajd_pham(covmats, sample_weight=w)
    assert_array_equal(V, Vw)
    assert_array_equal(D, Dw)


def test_uwedge():
    """Test uwedge."""
    covmats, diags, A = generate_cov(100, 3)
    V, D = uwedge(covmats)
    V, D = uwedge(covmats, init=A)
