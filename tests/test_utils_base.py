from functools import partial

import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from pytest import approx

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
    ddexpm,
    ddlogm,
)
from pyriemann.utils.test import is_pos_def, is_sym_pos_def, is_hermitian


n_channels = 3


def test_ctranspose(get_mats):
    X = np.random.rand(3, 4)
    assert_array_almost_equal(ctranspose(X), X.T, decimal=10)

    X = np.random.rand(7, 3, 4)
    assert_array_almost_equal(ctranspose(X), X.transpose(0, 2, 1), decimal=10)

    X = get_mats(10, n_channels, "herm")
    assert_array_almost_equal(ctranspose(X), X, decimal=10)


def test_expm():
    X = 2 * np.eye(n_channels)
    Xtrue = np.exp(2) * np.eye(n_channels)
    assert_array_almost_equal(expm(X), Xtrue, decimal=10)


def test_invsqrtm():
    X = 2 * np.eye(n_channels)
    Xtrue = (1.0 / np.sqrt(2)) * np.eye(n_channels)
    assert_array_almost_equal(invsqrtm(X), Xtrue, decimal=10)


def test_logm():
    X = 2 * np.eye(n_channels)
    Xtrue = np.log(2) * np.eye(n_channels)
    assert_array_almost_equal(logm(X), Xtrue, decimal=10)


def test_powm():
    X = 2 * np.eye(n_channels)
    Xtrue = (2 ** 0.3) * np.eye(n_channels)
    assert_array_almost_equal(powm(X, 0.3), Xtrue, decimal=10)


def test_sqrtm():
    X = 2 * np.eye(n_channels)
    Xtrue = np.sqrt(2) * np.eye(n_channels)
    assert_array_almost_equal(sqrtm(X), Xtrue, decimal=10)

    X = np.array([[1, -1j], [1j, 1]])
    Xtrue = np.sqrt(2) / 2 * X
    assert_array_almost_equal(sqrtm(X), Xtrue, decimal=10)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("funm", [
    expm,
    invsqrtm,
    logm,
    pytest.param(partial(powm, alpha=0.2), id="powm"),
    sqrtm
])
def test_funm_all(kind, funm, get_mats):
    n_matrices, n_dim = 10, 3
    X = get_mats(n_matrices, n_dim, kind)
    Xt = funm(X)
    assert Xt.shape == (n_matrices, n_dim, n_dim)


def test_funm_error():
    with pytest.raises(ValueError):
        sqrtm(np.ones(5))
    with pytest.raises(ValueError):
        invsqrtm(5.1)
    with pytest.raises(ValueError):
        logm([5.2])


@pytest.mark.parametrize("funm", [
    expm,
    invsqrtm,
    logm,
    pytest.param(partial(powm, alpha=0.2), id="powm"),
    sqrtm,
    nearest_sym_pos_def,
])
def test_funm_broadcasting(funm, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels = 2, 6, 5, 3
    X = get_mats([n_dim5, n_dim4, n_matrices], n_channels, "spd")

    # 2D array
    F2 = funm(X[0, 0, 0])
    assert F2.shape == (n_channels, n_channels)

    # 3D array
    F3 = funm(X[0, 0])
    assert F3.shape == (n_matrices, n_channels, n_channels)
    assert F3[0] == approx(F2)

    # 4D array
    F4 = funm(X[0])
    assert F4.shape == (n_dim4, n_matrices, n_channels, n_channels)
    assert F4[0, 0] == approx(F2)

    # 5D array
    F5 = funm(X)
    assert F5.shape == (n_dim5, n_dim4, n_matrices, n_channels, n_channels)
    assert F5[0, 0, 0] == approx(F2)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_funm_properties(get_mats, kind):
    n_matrices, n_dim = 10, 3
    X = get_mats(n_matrices, n_dim, kind)
    invX = np.linalg.solve(X, np.eye(n_dim))

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


def test_nearest_sym_pos_def(get_mats):
    n_matrices = 3
    X = get_mats(n_matrices, n_channels, "spd")
    D = X.diagonal(axis1=1, axis2=2)
    Psd = np.array([x - np.diag(d) for x, d in zip(X, D)])

    assert not is_pos_def(Psd)
    assert is_sym_pos_def(nearest_sym_pos_def(X))
    assert is_sym_pos_def(nearest_sym_pos_def(Psd))

    X = get_mats([4, 3, n_matrices], n_channels, "spd")
    nearest_sym_pos_def(X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_first_divided_difference(get_mats, kind):
    n_dim4, n_matrices = 7, 4
    X = get_mats([n_dim4, n_matrices], n_channels, kind)
    d = np.linalg.eigvalsh(X)

    fct, fctder = np.exp, np.exp
    fdd = _first_divided_difference(d, fct, fctder)

    assert fdd.shape == X.shape
    assert is_hermitian(fdd)
    np.allclose(np.diagonal(fdd, axis1=-2, axis2=-1), fctder(d))

    # test broadcasting
    assert fdd[0, 0] == approx(_first_divided_difference(d[0, 0], fct, fctder))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_first_divided_difference_properties(get_mats, kind):
    X = get_mats(1, n_channels, kind)[0]
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


@pytest.mark.parametrize("ddfun", [ddlogm, ddexpm])
def test_directional_derivative(ddfun, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels = 2, 6, 5, 3
    X = get_mats([n_dim5, n_dim4, n_matrices], n_channels, "sym")
    Cref = get_mats(1, n_channels, "spd")[0]

    DD = ddfun(X, Cref)
    assert DD.shape == X.shape

    # test broadcasting
    assert DD[0, 0] == approx(ddfun(X[0, 0], Cref))


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("ddfun", [ddlogm, ddexpm])
def test_directional_derivative_properties(kind, ddfun, get_mats, rndstate):
    X, Y, Cref = get_mats(3, n_channels, kind)
    Xdd = ddfun(X, Cref)

    # linearity
    a, b = rndstate.uniform(0.01, 0.99, size=2)
    Ydd = ddfun(Y, Cref)
    assert ddfun(a * X + b * Y, Cref) == approx(a * Xdd + b * Ydd)

    # self-adjointness wrt Frob inner product
    assert_array_almost_equal(np.trace(Xdd @ Y), np.trace(X @ Ydd))

    # identity reference
    Xdd = ddfun(X, np.eye(n_channels))
    if ddfun is ddexpm:
        assert_array_almost_equal(Xdd, np.exp(1) * X)
    elif ddfun is ddlogm:
        assert_array_almost_equal(Xdd, X)
