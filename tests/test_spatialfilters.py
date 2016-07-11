from nose.tools import assert_true, assert_raises
import numpy as np
from pyriemann.spatialfilters import Xdawn
from pyriemann.utils.covariance import _scm


def test_Xdawn():
    """Test Xdawn"""
    X = np.random.randn(100, 3, 10)
    y = np.array([0, 1]).repeat(50)
    xd = Xdawn()
    xd.fit(X, y)
    xd.transform(X)

    # 2. Test fit and transform Xdawn when the baseline covariance is
    # precomputed
    Cx = _scm(X.transpose(1, 2, 0).reshape(3, -1))
    # Error if baseline_cov isn't square matrix
    assert_raises(ValueError, Xdawn, baseline_cov='foo')

    Xt = Xdawn().fit_transform(X, y)
    # Error if baseline_cov doesn't have the same dimensionality as X
    assert_raises(ValueError, Xdawn(baseline_cov=Cx[:1, :1]).fit, X, y)
    # Same values if precomputed on identical data
    Xt_bsl = Xdawn(baseline_cov=Cx).fit_transform(X, y)
    np.testing.assert_array_equal(Xt, Xt_bsl)
    # Different values if precomputed on different data
    Cx = _scm(np.random.rand(3, 20))
    Xt_bsl = Xdawn(baseline_cov=Cx).fit_transform(X, y)
    assert_true(np.sum((Xt - Xt_bsl) ** 2))

    # 3. Test fit and transform Xdawn when some channels are flat
    X[:, 2, :] = 999  # half of the sensors are flat
    Xt = Xdawn().fit_transform(X, y)
    assert_true(np.sum(Xt ** 2) != 0)
    # check when baseline covariance doesn't have bad channels
    Cx = _scm(np.random.rand(3, 20))
    Xt = Xdawn(baseline_cov=Cx).fit_transform(X, y)
    assert_true(np.sum(Xt ** 2) != 0)
