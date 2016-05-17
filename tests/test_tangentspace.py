"""Test tangent space functions."""
import numpy as np
from pyriemann.tangentspace import TangentSpace, FGDA
from numpy.testing import assert_array_almost_equal
from nose.tools import assert_raises


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose."""
    np.random.seed(1234)
    diags = 2.0+0.1*np.random.randn(Nt, Ne)
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats


def test_TangentSpace_init():
    """Test init of Tangent."""
    TangentSpace(metric='riemann')


def test_TangentSpace_fit():
    """Test Fit of Tangent Space."""
    covset = generate_cov(10, 3)
    ts = TangentSpace(metric='riemann')
    ts.fit(covset)


def test_TangentSpace_transform():
    """Test transform of Tangent Space."""
    covset = generate_cov(10, 3)
    ts = TangentSpace(metric='riemann')
    ts.fit(covset)
    ts.transform(covset)

    X = np.zeros(shape=(10, 9))
    assert_raises(ValueError, ts.transform, X)

    X = np.zeros(shape=(10, 9, 8))
    assert_raises(ValueError, ts.transform, X)

    X = np.zeros(shape=(10))
    assert_raises(ValueError, ts.transform, X)

    X = np.zeros(shape=(12, 8, 8))
    assert_raises(ValueError, ts.transform, X)


def test_TangentSpace_transform_without_fit():
    """Test transform of Tangent Space without fit."""
    covset = generate_cov(10, 3)
    ts = TangentSpace(metric='riemann')
    ts.transform(covset)


def test_TangentSpace_transform_with_ts_update():
    """Test transform of Tangent Space with TSupdate."""
    covset = generate_cov(10, 3)
    ts = TangentSpace(metric='riemann', tsupdate=True)
    ts.fit(covset)
    ts.transform(covset)


def test_TangentSpace_inversetransform():
    """Test inverse transform of Tangent Space."""
    covset = generate_cov(10, 3)
    ts = TangentSpace(metric='riemann')
    ts.fit(covset)
    t = ts.transform(covset)
    cov = ts.inverse_transform(t)
    assert_array_almost_equal(covset, cov)


def test_TangentSpace_inversetransform_without_fit():
    """Test inverse transform of Tangent Space without fit."""
    Nt = 10
    Ne = 3 * 4 / 2
    tsv = np.random.randn(Nt, Ne)
    ts = TangentSpace(metric='riemann')
    ts.inverse_transform(tsv)


def test_FGDA_init():
    """Test init of FGDA."""
    FGDA(metric='riemann')


def test_FGDA_fit():
    """Test Fit of FGDA."""
    covset = generate_cov(10, 3)
    labels = np.array([0, 1]).repeat(5)
    ts = FGDA(metric='riemann')
    ts.fit(covset, labels)


def test_FGDA_transform():
    """Test transform of FGDA."""
    covset = generate_cov(10, 3)
    labels = np.array([0, 1]).repeat(5)
    ts = FGDA(metric='riemann')
    ts.fit(covset, labels)
    ts.transform(covset)
