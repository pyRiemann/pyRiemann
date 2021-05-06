import numpy as np
from numpy.testing import assert_array_equal
from pyriemann.spatialfilters import Xdawn, CSP, SPoC, BilinearFilter
import pytest


def generate_cov(Nt, Ne):
    """Generate a set of cavariances matrices for test purpose"""
    rs = np.random.RandomState(1234)
    diags = 2.0 + 0.1 * rs.randn(Nt, Ne)
    A = 2*rs.rand(Ne, Ne) - 1
    A /= np.atleast_2d(np.sqrt(np.sum(A**2, 1))).T
    covmats = np.empty((Nt, Ne, Ne))
    for i in range(Nt):
        covmats[i] = np.dot(np.dot(A, np.diag(diags[i])), A.T)
    return covmats


def test_Xdawn_init():
    """Test init of Xdawn"""
    xd = Xdawn()


def test_Xdawn_fit():
    """Test Fit of Xdawn"""
    x = np.random.randn(100, 3, 10)
    labels = np.array([0, 1]).repeat(50)
    xd = Xdawn()
    xd.fit(x, labels)


def test_Xdawn_transform():
    """Test transform of Xdawn"""
    x = np.random.randn(100, 3, 10)
    labels = np.array([0, 1]).repeat(50)
    xd = Xdawn()
    xd.fit(x, labels)
    xd.transform(x)


def test_Xdawn_baselinecov():
    """Test cov precomputation"""
    x = np.random.randn(100, 3, 10)
    labels = np.array([0, 1]).repeat(50)
    baseline_cov = np.identity(3)
    xd = Xdawn(baseline_cov=baseline_cov)
    xd.fit(x, labels)
    xd.transform(x)


def test_CSP():
    """Test CSP"""
    n_trials = 90
    X = generate_cov(n_trials, 3)
    labels = np.array([0, 1, 2]).repeat(n_trials // 3)

    # Test Init
    csp = CSP()
    assert csp.nfilter == 4
    assert csp.metric == 'euclid'
    assert csp.log
    csp = CSP(3, 'riemann', False)
    assert csp.nfilter == 3
    assert csp.metric == 'riemann'
    assert not csp.log

    with pytest.raises(TypeError):
        CSP('foo')

    with pytest.raises(ValueError):
        CSP(metric='foo')

    with pytest.raises(TypeError):
        CSP(log='foo')

    # Test fit
    csp = CSP()
    csp.fit(X, labels % 2)  # two classes
    csp.fit(X, labels)  # 3 classes

    with pytest.raises(ValueError):
        csp.fit(X, labels * 0.)  # 1 class
    with pytest.raises(ValueError):
        csp.fit(X, labels[:1])  # unequal # of samples
    with pytest.raises(TypeError):
        csp.fit(X, 'foo')  # y must be an array
    with pytest.raises(TypeError):
        csp.fit('foo', labels)  # X must be an array
    with pytest.raises(ValueError):
        csp.fit(X[:, 0], labels)
    with pytest.raises(ValueError):
        csp.fit(X, X)

    assert_array_equal(csp.filters_.shape, [X.shape[1], X.shape[1]])
    assert_array_equal(csp.patterns_.shape, [X.shape[1], X.shape[1]])

    # Test transform
    Xt = csp.transform(X)
    assert_array_equal(Xt.shape, [len(X), X.shape[1]])

    with pytest.raises(TypeError):
        csp.transform('foo')
    with pytest.raises(ValueError):
        csp.transform(X[:, 1:, :])  # unequal # of chans

    csp.log = False
    Xt = csp.transform(X)


def test_Spoc():
    """Test Spoc"""
    n_trials = 90
    X = generate_cov(n_trials, 3)
    labels = np.random.randn(n_trials)

    # Test Init
    spoc = SPoC()

    # Test fit
    spoc.fit(X, labels)


def test_BilinearFilter():
    """Test Bilinear filter"""
    n_trials = 90
    X = generate_cov(n_trials, 3)
    labels = np.array([0, 1, 2]).repeat(n_trials // 3)
    filters = np.eye(3)
    # Test Init
    bf = BilinearFilter(filters)
    assert not bf.log
    with pytest.raises(TypeError):
        BilinearFilter('foo')

    with pytest.raises(TypeError):
        BilinearFilter(np.eye(3), log='foo')

    # test fit
    bf = BilinearFilter(filters)
    bf.fit(X, labels % 2)

    # Test transform
    Xt = bf.transform(X)
    assert_array_equal(Xt.shape, [len(X), filters.shape[0], filters.shape[0]])

    with pytest.raises(TypeError):
        bf.transform('foo')
    with pytest.raises(ValueError):
        bf.transform(X[:, 1:, :])  # unequal # of chans

    bf.log = True
    Xt = bf.transform(X)
    assert_array_equal(Xt.shape, [len(X), filters.shape[0]])

    filters = filters[0:2, :]
    bf = BilinearFilter(filters)
    Xt = bf.transform(X)
    assert_array_equal(Xt.shape, [len(X), filters.shape[0], filters.shape[0]])
