import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

from pyriemann.datasets.simulated import _make_eyes
from pyriemann.utils.base import (
    ctranspose,
    expm,
    invsqrtm,
    logm,
    powm,
    sqrtm,
    nearest_sym_pos_def,
    _first_divided_difference,
    ddlogm,
    ddexpm
)
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.test import is_pos_def, is_sym_pos_def


n_channels = 3


def test_ctranspose(get_mats):
    X = np.random.rand(3, 4)
    assert_array_almost_equal(ctranspose(X), X.T, decimal=10)

    X = np.random.rand(7, 3, 4)
    assert_array_almost_equal(ctranspose(X), X.transpose(0, 2, 1), decimal=10)

    X = get_mats(10, n_channels, "herm")
    assert_array_almost_equal(ctranspose(X), X, decimal=10)


def test_expm():
    """Test matrix exponential"""
    X = 2 * np.eye(n_channels)
    Xtrue = np.exp(2) * np.eye(n_channels)
    assert_array_almost_equal(expm(X), Xtrue, decimal=10)


def test_invsqrtm():
    """Test matrix inverse square root"""
    X = 2 * np.eye(n_channels)
    Xtrue = (1.0 / np.sqrt(2)) * np.eye(n_channels)
    assert_array_almost_equal(invsqrtm(X), Xtrue, decimal=10)


def test_logm():
    """Test matrix logarithm"""
    X = 2 * np.eye(n_channels)
    Xtrue = np.log(2) * np.eye(n_channels)
    assert_array_almost_equal(logm(X), Xtrue, decimal=10)


def test_powm():
    """Test matrix power"""
    X = 2 * np.eye(n_channels)
    Xtrue = (2 ** 0.3) * np.eye(n_channels)
    assert_array_almost_equal(powm(X, 0.3), Xtrue, decimal=10)


def test_sqrtm():
    """Test matrix square root"""
    X = 2 * np.eye(n_channels)
    Xtrue = np.sqrt(2) * np.eye(n_channels)
    assert_array_almost_equal(sqrtm(X), Xtrue, decimal=10)

    X = np.array([[1, -1j], [1j, 1]])
    Xtrue = np.sqrt(2) / 2 * X
    assert_array_almost_equal(sqrtm(X), Xtrue, decimal=10)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("funm", [expm, invsqrtm, logm, powm, sqrtm])
def test_funm_all(kind, funm, get_mats):
    n_matrices, n_dim = 10, 3
    X = get_mats(n_matrices, n_dim, kind)
    if funm is powm:
        Xt = funm(X, 0.42)
    else:
        Xt = funm(X)
    assert Xt.shape == (n_matrices, n_dim, n_dim)


def test_funm_error():
    with pytest.raises(ValueError):
        sqrtm(np.ones(5))
    with pytest.raises(ValueError):
        invsqrtm(5.1)
    with pytest.raises(ValueError):
        logm([5.2])


@pytest.mark.parametrize("funm", [expm, invsqrtm, logm, powm, sqrtm])
def test_funm_ndarray(funm):
    def test(funm, X):
        if funm == powm:
            Xt = funm(X, 0.2)
        else:
            Xt = funm(X)
        assert X.shape == Xt.shape

    n_matrices = 6
    X_3d = np.asarray([np.eye(n_channels) for _ in range(n_matrices)])
    test(funm, X_3d)

    n_sets = 5
    X_4d = np.asarray([X_3d for _ in range(n_sets)])
    test(funm, X_4d)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_funm_properties(get_mats, kind):
    n_matrices, n_dim = 10, 3
    X = get_mats(n_matrices, n_dim, kind)
    invX = np.linalg.inv(X)

    # expm and logm
    eX, lX = expm(X), logm(X)
    assert_array_almost_equal(eX.conj(), expm(X.conj()), decimal=10)
    assert_array_almost_equal(
        np.linalg.det(eX),
        np.exp(np.trace(X, axis1=-2, axis2=-1)),
        decimal=10,
    )
    assert_array_almost_equal(logm(eX), X, decimal=10)
    assert_array_almost_equal(expm(lX), X, decimal=10)
    assert_array_almost_equal(expm(-lX), invX, decimal=10)

    # invsqrtm
    isX = invsqrtm(X)
    eyes = _make_eyes(n_matrices, n_dim)
    assert_array_almost_equal(isX @ X @ isX, eyes, decimal=10)
    assert_array_almost_equal(isX @ isX, invX, decimal=10)

    # sqrtm
    sX = sqrtm(X)
    assert_array_almost_equal(ctranspose(sX) @ sX, X, decimal=10)
    assert_array_almost_equal(isX @ X @ isX, eyes, decimal=10)

    # powm
    assert_array_almost_equal(powm(X, 0.5), sX, decimal=10)
    assert_array_almost_equal(powm(X, -0.5), isX, decimal=10)
    alpha = 0.3
    assert_array_almost_equal(
        powm(X, alpha=alpha),
        expm(alpha * logm(X)),
        decimal=10,
    )


def test_check_raise():
    """Test check SPD matrices"""
    X = 2 * np.ones((10, n_channels, n_channels))
    # This is an indirect check, the riemannian mean must crash when the
    # matrices are not SPD.
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError):
            mean_riemann(X)


def test_nearest_sym_pos_def(get_mats):
    n_matrices = 3
    X = get_mats(n_matrices, n_channels, "spd")
    D = X.diagonal(axis1=1, axis2=2)
    Psd = np.array([x - np.diag(d) for x, d in zip(X, D)])

    assert not is_pos_def(Psd)
    assert is_sym_pos_def(nearest_sym_pos_def(X))
    assert is_sym_pos_def(nearest_sym_pos_def(Psd))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_first_divided_difference(get_mats, kind):
    """Test first divided difference."""
    n_matrices = 1
    X = get_mats(n_matrices, n_channels, kind)[0]
    d = np.linalg.eigvalsh(X)

    fdd_id = _first_divided_difference(d, lambda x: x, lambda x: x)
    assert fdd_id.shape == X.shape
    assert_array_almost_equal(np.diag(fdd_id), d)
    assert_array_almost_equal(fdd_id[np.triu_indices_from(fdd_id, k=1)], 1)

    fdd_exp = _first_divided_difference(d, np.exp, np.exp)
    assert_array_almost_equal(np.diag(fdd_exp), np.exp(d))

    fdd_log = _first_divided_difference(d, np.log, lambda x: 1./x)
    assert_array_almost_equal(np.diag(fdd_log), 1/d)

    # exp of log is element-wise inverse of log
    fdd_exp_of_log = _first_divided_difference(np.log(d), np.exp, np.exp)
    assert_array_almost_equal(fdd_exp_of_log, 1/fdd_log)


def test_ddlogm(get_mats):
    """Test directional derivative of log."""
    X, Cref = get_mats(2, n_channels, "spd")
    fdd_logm = ddlogm(X, Cref)
    assert fdd_logm.shape == X.shape

    fdd_logm = ddlogm(X, np.eye(n_channels))
    assert_array_almost_equal(fdd_logm, X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ddexpm(get_mats, kind):
    """Test directional derivative of exp."""
    X, Cref = get_mats(2, n_channels, kind)
    fdd_expm = ddexpm(X, Cref)
    assert fdd_expm.shape == X.shape

    fdd_expm = ddexpm(X, np.eye(n_channels))
    assert_array_almost_equal(fdd_expm, np.exp(1)*X)
