import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest

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
    C = 2 * np.eye(n_channels)
    Ctrue = np.exp(2) * np.eye(n_channels)
    assert_array_almost_equal(expm(C), Ctrue, decimal=10)


def test_invsqrtm():
    """Test matrix inverse square root"""
    C = 2 * np.eye(n_channels)
    Ctrue = (1.0 / np.sqrt(2)) * np.eye(n_channels)
    assert_array_almost_equal(invsqrtm(C), Ctrue, decimal=10)


def test_logm():
    """Test matrix logarithm"""
    C = 2 * np.eye(n_channels)
    Ctrue = np.log(2) * np.eye(n_channels)
    assert_array_almost_equal(logm(C), Ctrue, decimal=10)


def test_powm():
    """Test matrix power"""
    C = 2 * np.eye(n_channels)
    Ctrue = (2 ** 0.3) * np.eye(n_channels)
    assert_array_almost_equal(powm(C, 0.3), Ctrue, decimal=10)


def test_sqrtm():
    """Test matrix square root"""
    C = 2 * np.eye(n_channels)
    Ctrue = np.sqrt(2) * np.eye(n_channels)
    assert_array_almost_equal(sqrtm(C), Ctrue, decimal=10)

    C = np.array([[1, -1j], [1j, 1]])
    Ctrue = np.sqrt(2) / 2 * C
    assert_array_almost_equal(sqrtm(C), Ctrue, decimal=10)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("funm", [expm, invsqrtm, logm, powm, sqrtm])
def test_funm_all(kind, funm, get_mats):
    n_matrices, n_dim = 10, 3
    C = get_mats(n_matrices, n_dim, kind)
    if funm is powm:
        D = funm(C, 0.42)
    else:
        D = funm(C)
    assert D.shape == (n_matrices, n_dim, n_dim)


def test_funm_error():
    with pytest.raises(ValueError):
        sqrtm(np.ones(5))
    with pytest.raises(ValueError):
        invsqrtm(5.1)
    with pytest.raises(ValueError):
        logm([5.2])


@pytest.mark.parametrize("funm", [expm, invsqrtm, logm, powm, sqrtm])
def test_funm_ndarray(funm):
    def test(funm, C):
        if funm == powm:
            D = funm(C, 0.2)
        else:
            D = funm(C)
        assert C.shape == D.shape

    n_matrices = 6
    C_3d = np.asarray([np.eye(n_channels) for _ in range(n_matrices)])
    test(funm, C_3d)

    n_sets = 5
    C_4d = np.asarray([C_3d for _ in range(n_sets)])
    test(funm, C_4d)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_funm_properties(get_mats, kind):
    n_matrices, n_dim = 10, 3
    C = get_mats(n_matrices, n_dim, kind)
    invC = np.linalg.inv(C)

    # expm and logm
    eC, lC = expm(C), logm(C)
    assert_array_almost_equal(eC.conj(), expm(C.conj()), decimal=10)
    assert_array_almost_equal(
        np.linalg.det(eC),
        np.exp(np.trace(C, axis1=-2, axis2=-1)),
        decimal=10,
    )
    assert_array_almost_equal(logm(eC), C, decimal=10)
    assert_array_almost_equal(expm(lC), C, decimal=10)
    assert_array_almost_equal(expm(-lC), invC, decimal=10)

    # invsqrtm
    isC = invsqrtm(C)
    Eye = np.repeat(np.eye(n_dim)[np.newaxis, :, :], n_matrices, axis=0)
    assert_array_almost_equal(isC @ C @ isC, Eye, decimal=10)
    assert_array_almost_equal(isC @ isC, invC, decimal=10)

    # sqrtm
    sC = sqrtm(C)
    assert_array_almost_equal(ctranspose(sC) @ sC, C, decimal=10)
    assert_array_almost_equal(isC @ C @ isC, Eye, decimal=10)

    # powm
    assert_array_almost_equal(powm(C, 0.5), sC, decimal=10)
    assert_array_almost_equal(powm(C, -0.5), isC, decimal=10)
    alpha = 0.3
    assert_array_almost_equal(
        powm(C, alpha=alpha),
        expm(alpha * logm(C)),
        decimal=10,
    )


def test_check_raise():
    """Test check SPD matrices"""
    C = 2 * np.ones((10, n_channels, n_channels))
    # This is an indirect check, the riemannian mean must crash when the
    # matrices are not SPD.
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError):
            mean_riemann(C)


def test_nearest_sym_pos_def(get_mats):
    n_matrices = 3
    mats = get_mats(n_matrices, n_channels, "spd")
    D = mats.diagonal(axis1=1, axis2=2)
    psd = np.array([mat - np.diag(d) for mat, d in zip(mats, D)])

    assert not is_pos_def(psd)
    assert is_sym_pos_def(nearest_sym_pos_def(mats))
    assert is_sym_pos_def(nearest_sym_pos_def(psd))


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
    n_matrices = 2
    X, Cref = get_mats(n_matrices, n_channels, "spd")
    fdd_logm = ddlogm(X, Cref)
    assert fdd_logm.shape == X.shape

    fdd_logm = ddlogm(X, np.eye(n_channels))
    assert_array_almost_equal(fdd_logm, X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ddexpm(get_mats, kind):
    """Test directional derivative of exp."""
    n_matrices = 2
    X, Cref = get_mats(n_matrices, n_channels, kind)
    fdd_expm = ddexpm(X, Cref)
    assert fdd_expm.shape == X.shape

    fdd_expm = ddexpm(X, np.eye(n_channels))
    assert_array_almost_equal(fdd_expm, np.exp(1)*X)
