import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

from pyriemann.utils.base import (
    sqrtm,
    invsqrtm,
    logm,
    expm,
    powm,
    nearest_pos_def,
)
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.test import is_pos_def


n_channels = 3


def test_sqrtm():
    """Test matrix square root"""
    C = 2 * np.eye(n_channels)
    Ctrue = np.sqrt(2) * np.eye(n_channels)
    assert_array_almost_equal(sqrtm(C), Ctrue)


def test_invsqrtm():
    """Test matrix inverse square root"""
    C = 2 * np.eye(n_channels)
    Ctrue = (1.0 / np.sqrt(2)) * np.eye(n_channels)
    assert_array_almost_equal(invsqrtm(C), Ctrue)


def test_logm():
    """Test matrix logarithm"""
    C = 2 * np.eye(n_channels)
    Ctrue = np.log(2) * np.eye(n_channels)
    assert_array_almost_equal(logm(C), Ctrue)


def test_expm():
    """Test matrix exponential"""
    C = 2 * np.eye(n_channels)
    Ctrue = np.exp(2) * np.eye(n_channels)
    assert_array_almost_equal(expm(C), Ctrue)


def test_powm():
    """Test matrix power"""
    C = 2 * np.eye(n_channels)
    Ctrue = (2 ** 0.5) * np.eye(n_channels)
    assert_array_almost_equal(powm(C, 0.5), Ctrue)


def test_check_raise():
    """Test check SPD matrices"""
    C = 2 * np.ones((10, n_channels, n_channels))
    # This is an indirect check, the riemannian mean must crash when the
    # matrices are not SPD.
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError):
            mean_riemann(C)


def test_funm_error():
    with pytest.raises(ValueError):
        sqrtm(np.ones(5))
    with pytest.raises(ValueError):
        invsqrtm(5.1)
    with pytest.raises(ValueError):
        logm([5.2])


@pytest.mark.parametrize("funm", [sqrtm, invsqrtm, logm, expm, powm])
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


def test_nearest_pos_def(get_covmats):
    n_matrices = 3
    covmats = get_covmats(n_matrices, n_channels)
    D = covmats.diagonal(axis1=1, axis2=2)
    psd = np.array([cov - np.diag(d) for cov, d in zip(covmats, D)])

    assert is_pos_def(nearest_pos_def(covmats))
    assert not is_pos_def(psd)
    assert is_pos_def(nearest_pos_def(psd))
