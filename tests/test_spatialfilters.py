import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true, assert_raises
from pyriemann.spatialfilters import Xdawn, CSP


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
    n_trials = 100
    X = np.array([np.identity(3)] * n_trials)
    labels = np.array([0, 1]).repeat(n_trials // 2)

    # Test Init
    csp = CSP()
    assert_true(csp.nfilter == 4)
    assert_true(csp.metric == 'euclid')
    assert_true(csp.log)
    csp = CSP(3, 'riemann', False)
    assert_true(csp.nfilter == 3)
    assert_true(csp.metric == 'riemann')
    assert_true(not csp.log)
    assert_raises(TypeError, CSP, 'foo')
    assert_raises(ValueError, CSP, metric='foo')

    # Test fit
    csp = CSP()
    csp.fit(X, labels % 2)  # two classes
    csp.fit(X, labels)  # 3 classes
    assert_raises(ValueError, csp.fit, X, labels * 0.)  # 1 class
    assert_raises(ValueError, csp.fit, X, labels[:1])  # unequal # of samples
    assert_raises(TypeError, csp.fit, X, 'foo')
    assert_array_equal(csp.filters_.shape, [X.shape[1], X.shape[1]])
    assert_array_equal(csp.patterns_.shape, [X.shape[1], X.shape[1]])

    # Test transform
    Xt = csp.transform(X)
    assert_array_equal(Xt.shape, [len(X), X.shape[1]])
    assert_raises(TypeError, csp.transform, 'foo')
    assert_raises(ValueError, csp.transform, X[:, 1:, :])  # unequal # of chans
