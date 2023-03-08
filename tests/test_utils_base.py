import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from pyriemann.utils.base import (
    expm,
    invsqrtm,
    logm,
    powm,
    sqrtm,
    nearest_sym_pos_def,
)
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.test import is_pos_def, is_sym_pos_def


n_channels = 3


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
    assert_array_almost_equal(
        np.swapaxes(sC.conj(), -2, -1) @ sC,
        C,
        decimal=10,
    )
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


def test_nearest_sym_pos_def(get_covmats):
    n_matrices = 3
    mats = get_covmats(n_matrices, n_channels)
    D = mats.diagonal(axis1=1, axis2=2)
    psd = np.array([mat - np.diag(d) for mat, d in zip(mats, D)])

    assert not is_pos_def(psd)
    assert is_sym_pos_def(nearest_sym_pos_def(mats))
    assert is_sym_pos_def(nearest_sym_pos_def(psd))
