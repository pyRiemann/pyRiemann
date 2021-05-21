from numpy.testing import assert_array_equal
import numpy as np
import pytest

from pyriemann.utils.ajd import rjd, ajd_pham, uwedge, _get_normalized_weight


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


def test_get_normalized_weight():
    """Test get_normalized_weight"""
    Nt = 100
    covmats, diags, A = generate_cov(Nt, 3)
    w = _get_normalized_weight(None, covmats)
    assert np.isclose(np.sum(w), 1., atol=1e-10)

    with pytest.raises(ValueError): # not same length
        _get_normalized_weight(w[:Nt//2], covmats)
    with pytest.raises(ValueError): # not strictly positive weight
        w[0] = 0
        _get_normalized_weight(w, covmats)


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
    assert_array_equal(V, Vw) # same result as ajd_pham without weight
    assert_array_equal(D, Dw)

    with pytest.raises(ValueError): # not strictly positive weight
        w[0] = 0
        ajd_pham(covmats, sample_weight=w)


def test_uwedge():
    """Test uwedge."""
    covmats, diags, A = generate_cov(100, 3)
    V, D = uwedge(covmats)
    V, D = uwedge(covmats, init=A)
