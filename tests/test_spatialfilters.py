import numpy as np
from pyriemann.spatialfilters import Xdawn, CSP
from nose.tools import assert_raises


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose"""
    diags = 1.0+0.1*np.random.randn(Nt, Ne)
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.diag(diags[i])
    return covmats


def test_Xdawn_fit():
    """Test Fit of Xdawn"""
    X = np.random.randn(100, 3, 10)
    y = np.array([0, 1]).repeat(50)
    xd = Xdawn()
    xd.fit(X, y)

    xd = Xdawn(n_filters=2, classes=[0, 1], estimator='lwf')
    y = np.array([0, 1, 2, 3]).repeat(25)
    xd.fit(X, y)


def test_Xdawn_transform():
    """Test transform of Xdawn"""
    X = np.random.randn(100, 3, 10)
    y = np.array([0, 1]).repeat(50)
    xd = Xdawn()
    xd.fit(X, y)
    xd.transform(X)


def test_CSP():
    """Test methods of CSP"""
    csp = CSP()
    X = covset = generate_cov(100, 3)
    y = np.array([0, 1]).repeat(50)
    csp.fit(X, y)
    csp.transform(X)

    csp = CSP(n_filters=2, metric='logeuclid')
    csp.fit(X, y)
    csp.transform(X)

    csp = CSP()
    y = np.zeros(shape=(100,))
    assert_raises(ValueError, csp.fit, X, y)
    

def test_CSP_multiclass():
    """Test multiclass CSP"""
    csp = CSP()
    X = covset = generate_cov(100, 3)
    y = np.array([0, 1, 2, 3]).repeat(25)
    csp.fit(X, y)
    csp.transform(X)
