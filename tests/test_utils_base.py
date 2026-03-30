from functools import partial
import warnings

import numpy as np
from conftest import assert_array_almost_equal, to_numpy
import pytest
from pytest import approx

from pyriemann.utils._backend import get_namespace, xpd as device
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
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.geodesic import geodesic_logchol
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import log_map_riemann, exp_map_riemann
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
    with pytest.raises((ValueError, TypeError)):
        sqrtm(np.ones(5))
    with pytest.raises((ValueError, TypeError)):
        invsqrtm(5.1)
    with pytest.raises((ValueError, TypeError)):
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
    xp = get_namespace(X)
    invX = xp.linalg.inv(X)

    # expm and logm
    eX, lX = expm(X), logm(X)
    assert_array_almost_equal(eX.conj(), expm(X.conj()), decimal=10)
    assert_array_almost_equal(
        to_numpy(xp.linalg.det(eX)),
        to_numpy(xp.exp(xp.sum(
            X * xp.eye(n_dim, dtype=X.dtype, device=device(X)),
            axis=(-2, -1),
        ))),
        decimal=10,
    )
    assert_array_almost_equal(logm(eX), X, decimal=10)
    assert_array_almost_equal(expm(lX), X, decimal=10)
    assert_array_almost_equal(expm(-lX), invX, decimal=10)

    # invsqrtm
    isX = invsqrtm(X)
    eyes = xp.broadcast_to(
        xp.eye(n_dim, dtype=X.dtype, device=device(X)),
        (n_matrices, n_dim, n_dim),
    )
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


@pytest.mark.numpy_only
def test_check_raise():
    """Test check SPD matrices"""
    from pyriemann.utils.mean import mean_riemann
    X = 2 * np.ones((10, n_channels, n_channels))
    # This is an indirect check, the riemannian mean must crash when the
    # matrices are not SPD.
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError):
            mean_riemann(X)


def test_nearest_sym_pos_def(get_mats):
    n_matrices = 3
    X = get_mats(n_matrices, n_channels, "spd")
    X_np = to_numpy(X)
    D = X_np.diagonal(axis1=1, axis2=2)
    Psd = np.array([x - np.diag(d) for x, d in zip(X_np, D)])

    assert not is_pos_def(Psd)
    assert is_sym_pos_def(nearest_sym_pos_def(X_np))
    assert is_sym_pos_def(nearest_sym_pos_def(Psd))

    # test with backend arrays and broadcasting
    assert is_sym_pos_def(to_numpy(nearest_sym_pos_def(X)))
    X_5d = get_mats([4, 3, n_matrices], n_channels, "spd")
    nearest_sym_pos_def(X_5d)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_first_divided_difference(get_mats, kind):
    """Test first divided difference."""
    n_matrices = 1
    X = get_mats(n_matrices, n_channels, kind)[0]
    xp = get_namespace(X)
    d = xp.linalg.eigvalsh(X)

    fdd_id = _first_divided_difference(d, lambda x: x, lambda x: x)
    assert fdd_id.shape == X.shape
    d_np = to_numpy(d)
    fdd_id_np = to_numpy(fdd_id)
    assert_array_almost_equal(np.diag(fdd_id_np), d_np)
    assert_array_almost_equal(
        fdd_id_np[np.triu_indices_from(fdd_id_np, k=1)], 1
    )

    fdd_exp = _first_divided_difference(d, xp.exp, xp.exp)
    assert_array_almost_equal(
        np.diag(to_numpy(fdd_exp)), np.exp(d_np)
    )

    fdd_log = _first_divided_difference(d, xp.log, lambda x: 1./x)
    assert_array_almost_equal(np.diag(to_numpy(fdd_log)), 1/d_np)

    # exp of log is element-wise inverse of log
    fdd_exp_of_log = _first_divided_difference(
        xp.log(d), xp.exp, xp.exp
    )
    assert_array_almost_equal(fdd_exp_of_log, 1/to_numpy(fdd_log))


def test_ddlogm(get_mats):
    """Test directional derivative of log."""
    X, Cref = get_mats(2, n_channels, "spd")
    xp = get_namespace(X)
    fdd_logm = ddlogm(X, Cref)
    assert fdd_logm.shape == X.shape

    eye = xp.eye(n_channels, dtype=X.dtype, device=device(X))
    fdd_logm = ddlogm(X, eye)
    assert_array_almost_equal(fdd_logm, X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
def test_ddexpm(get_mats, kind):
    """Test directional derivative of exp."""
    X, Cref = get_mats(2, n_channels, kind)
    xp = get_namespace(X)
    fdd_expm = ddexpm(X, Cref)
    assert fdd_expm.shape == X.shape

    eye = xp.eye(n_channels, dtype=X.dtype, device=device(X))
    fdd_expm = ddexpm(X, eye)
    assert_array_almost_equal(fdd_expm, np.exp(1) * to_numpy(X))


@pytest.mark.numpy_only
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


def test_autograd_smoke():
    torch = pytest.importorskip("torch")
    torch.set_grad_enabled(True)  # re-enable for this test
    rng = np.random.RandomState(199)

    def _make_spd(shape, n):
        mats = rng.standard_normal((*shape, n, n))
        return mats @ np.swapaxes(mats, -1, -2) + 0.25 * np.eye(n)

    def _to_torch(x):
        return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64)

    A = _to_torch(_make_spd((), 3)).clone().detach().requires_grad_(True)
    B = _to_torch(_make_spd((), 3)).clone().detach().requires_grad_(True)
    X = _to_torch(_make_spd((4,), 3)).clone().detach().requires_grad_(True)

    tangent = log_map_riemann(B.unsqueeze(0), A, C12=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss = distance_riemann(A, B)
        loss = loss + geodesic_logchol(A, B, alpha=0.25).sum()
        loss = loss + mean_riemann(X, maxiter=5).sum()
        loss = loss + exp_map_riemann(tangent, A, Cm12=True).sum()
    loss.backward()

    for grad in (A.grad, B.grad, X.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
