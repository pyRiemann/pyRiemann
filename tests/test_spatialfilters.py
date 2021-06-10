import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pyriemann.spatialfilters import (Whitening, Xdawn, CSP, SPoC,
                                      BilinearFilter, AJDC)
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


def test_whitening():
    """Test Whitening"""
    n_trials, n_channels, n_components = 20, 6, 3
    cov = generate_cov(n_trials, n_channels)
    weights = np.random.rand(n_trials)
    max_cond = 10

    # Test Init
    whit = Whitening()
    assert whit.metric=='euclid'
    assert whit.dim_red==None
    assert whit.verbose==False

    # Test Fit
    with pytest.raises(ValueError): # len dim_red not equal to 1
        Whitening(dim_red={'n_components': 2, 'expl_var': 0.5}).fit(cov)
    with pytest.raises(ValueError): # n_components not superior to 1
        Whitening(dim_red={'n_components': 0}).fit(cov)
    with pytest.raises(ValueError): # n_components not a int
        Whitening(dim_red={'n_components': 2.5}).fit(cov)
    with pytest.raises(ValueError): # expl_var out of bound
        Whitening(dim_red={'expl_var': 0}).fit(cov)
    with pytest.raises(ValueError): # expl_var out of bound
        Whitening(dim_red={'expl_var': 1.1}).fit(cov)
    with pytest.raises(ValueError): # max_cond not strictly superior to 1
        Whitening(dim_red={'max_cond': 1}).fit(cov)
    with pytest.raises(ValueError): # unknown key
        Whitening(dim_red={'abc': 42}).fit(cov)
    with pytest.raises(ValueError): # unknown type
        Whitening(dim_red='max_cond').fit(cov)
    with pytest.raises(ValueError): # unknown type
        Whitening(dim_red=20).fit(cov)

    whit = Whitening().fit(cov)
    assert whit.n_components_ == n_channels
    assert_array_equal(whit.filters_.shape, [n_channels, n_channels])
    assert_array_equal(whit.inv_filters_.shape, [n_channels, n_channels])

    whit = Whitening(dim_red={'n_components': n_components}).fit(cov, weights)
    assert whit.n_components_ == n_components
    assert_array_equal(whit.filters_.shape, [n_channels, n_components])
    assert_array_equal(whit.inv_filters_.shape, [n_components, n_channels])

    whit = Whitening(dim_red={'expl_var': 0.9}).fit(cov)
    assert whit.n_components_ <= n_channels
    assert_array_equal(whit.filters_.shape, [n_channels, whit.n_components_])
    assert_array_equal(whit.inv_filters_.shape,
                       [whit.n_components_, n_channels])

    whit = Whitening(dim_red={'max_cond': max_cond}).fit(cov, weights)
    assert whit.n_components_ <= n_channels
    assert_array_equal(whit.filters_.shape, [n_channels, whit.n_components_])
    assert_array_equal(whit.inv_filters_.shape,
                       [whit.n_components_, n_channels])

    # Test transform
    whit = Whitening().fit(cov)
    cov_w = whit.transform(cov)
    assert_array_equal(cov_w.shape, [n_trials, n_channels, n_channels])
    # after whitening, mean = identity
    assert_array_almost_equal(cov_w.mean(axis=0), np.eye(n_channels))

    whit = Whitening(dim_red={'n_components': n_components}).fit(cov)
    cov_w = whit.transform(cov)
    n_components_ = whit.n_components_
    assert_array_equal(cov_w.shape, [n_trials, n_components_, n_components_])
    # after whitening, mean = identity
    assert_array_almost_equal(cov_w.mean(axis=0), np.eye(n_components_))

    whit = Whitening(dim_red={'expl_var': 0.9}).fit(cov)
    cov_w = whit.transform(cov)
    n_components_ = whit.n_components_
    assert_array_equal(cov_w.shape, [n_trials, n_components_, n_components_])
    # after whitening, mean = identity
    assert_array_almost_equal(cov_w.mean(axis=0), np.eye(n_components_))

    whit = Whitening(dim_red={'max_cond': max_cond}).fit(cov)
    cov_w = whit.transform(cov)
    n_components_ = whit.n_components_
    assert_array_equal(cov_w.shape, [n_trials, n_components_, n_components_])
    # after whitening, mean = identity
    mean = cov_w.mean(axis=0)
    assert_array_almost_equal(mean, np.eye(n_components_))
    assert np.linalg.cond(mean) <= max_cond

    # Test inverse_transform
    whit = Whitening().fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert_array_equal(cov_iw.shape, [n_trials, n_channels, n_channels])
    assert_array_almost_equal(cov, cov_iw)

    whit = Whitening(dim_red={'n_components': n_components}).fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert_array_equal(cov_iw.shape, [n_trials, n_channels, n_channels])

    whit = Whitening(dim_red={'expl_var': 0.9}).fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert_array_equal(cov_iw.shape, [n_trials, n_channels, n_channels])

    whit = Whitening(dim_red={'max_cond': max_cond}).fit(cov)
    cov_iw = whit.inverse_transform(whit.transform(cov))
    assert_array_equal(cov_iw.shape, [n_trials, n_channels, n_channels])


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


def test_AJDC():
    """Test AJDC"""
    rs = np.random.RandomState(33)
    n_subjects, n_conditions, n_channels, n_samples = 5, 3, 8, 512
    X = rs.randn(n_subjects, n_conditions, n_channels, n_samples)

    # Test Init
    ajdc = AJDC(fmin=1, fmax=32, fs=64)
    assert ajdc.window == 128
    assert ajdc.overlap == 0.5
    assert ajdc.dim_red == None
    assert ajdc.verbose

    # Test fit
    ajdc.fit(X)
    assert ajdc.n_channels_ == n_channels
    assert ajdc.n_sources_ <= n_channels
    assert_array_equal(ajdc.forward_filters_.shape,
                       [ajdc.n_sources_, n_channels])
    assert_array_equal(ajdc.backward_filters_.shape,
                       [n_channels, ajdc.n_sources_])
    with pytest.raises(ValueError): # unequal # of conditions
        ajdc.fit([rs.randn(n_conditions, n_channels, n_samples),
                  rs.randn(n_conditions + 1, n_channels, n_samples)])
    with pytest.raises(ValueError): # unequal # of channels
        ajdc.fit([rs.randn(n_conditions, n_channels, n_samples),
                  rs.randn(n_conditions, n_channels + 1, n_samples)])
    # 3 subjects, same # conditions and channels, different # of samples
    X = [rs.randn(n_conditions, n_channels, n_samples),
         rs.randn(n_conditions, n_channels, n_samples + 200),
         rs.randn(n_conditions, n_channels, n_samples + 500)]
    ajdc.fit(X)
    # 2 subjects, 2 conditions, same # channels, different # of samples
    X = [[rs.randn(n_channels, n_samples),
          rs.randn(n_channels, n_samples + 200)],
         [rs.randn(n_channels, n_samples + 500),
          rs.randn(n_channels, n_samples + 100)]]
    ajdc.fit(X)

    # Test transform
    n_trials = 4
    X = rs.randn(n_trials, n_channels, n_samples)
    Xt = ajdc.transform(X)
    assert_array_equal(Xt.shape, [n_trials, ajdc.n_sources_, n_samples])
    with pytest.raises(ValueError): # not 3 dims
        ajdc.transform(X[0])
    with pytest.raises(ValueError): # unequal # of chans
        ajdc.transform(rs.randn(n_trials, n_channels + 1, 1))

    # Test inverse_transform
    Xtb = ajdc.inverse_transform(Xt)
    assert_array_equal(Xtb.shape, [n_trials, n_channels, n_samples])
    with pytest.raises(ValueError): # not 3 dims
        ajdc.inverse_transform(Xt[0])
    with pytest.raises(ValueError): # unequal # of sources
        ajdc.inverse_transform(rs.randn(n_trials, ajdc.n_sources_ + 1, 1))

    Xtb = ajdc.inverse_transform(Xt, supp=[ajdc.n_sources_ - 1])
    assert_array_equal(Xtb.shape, [n_trials, n_channels, n_samples])
    with pytest.raises(ValueError): # not a list
        ajdc.inverse_transform(Xt, supp=1)

    # Test get_src_expl_var
    v = ajdc.get_src_expl_var(X)
    assert_array_equal(v.shape, [n_trials, ajdc.n_sources_])
    with pytest.raises(ValueError): # not 3 dims
        ajdc.get_src_expl_var(X[0])
    with pytest.raises(ValueError): # unequal # of chans
        ajdc.get_src_expl_var(rs.randn(n_trials, n_channels + 1, 1))
