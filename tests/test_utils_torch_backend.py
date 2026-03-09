import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.utils.distance import (  # noqa: E402
    distance_chol,
    distance_euclid,
    distance_harmonic,
    distance_logchol,
    distance_logeuclid,
    distance_riemann,
    distance_wasserstein,
    distance,
    pairwise_distance,
)
from pyriemann.utils.base import sqrtm  # noqa: E402
from pyriemann.utils.geodesic import (  # noqa: E402
    geodesic_chol,
    geodesic_logchol,
    geodesic_logeuclid,
    geodesic_riemann,
    geodesic_wasserstein,
)
from pyriemann.utils.mean import (  # noqa: E402
    mean_chol,
    mean_harmonic,
    mean_logchol,
    mean_logeuclid,
    mean_riemann,
    mean_wasserstein,
)
from pyriemann.utils.tangentspace import (  # noqa: E402
    exp_map_riemann,
    exp_map_logchol,
    log_map_riemann,
    log_map_logchol,
    tangent_space,
    untangent_space,
    upper,
    unupper,
)


STRICT_TOL = dict(atol=1e-7, rtol=1e-6)
ITER_TOL = dict(atol=1e-6, rtol=1e-5)


def _make_spd(batch_shape, n, rng):
    mats = rng.standard_normal((*batch_shape, n, n))
    return mats @ np.swapaxes(mats, -1, -2) + 0.25 * np.eye(n)


def _to_torch(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64)


@pytest.mark.parametrize(
    "fn",
    [
        distance_chol,
        distance_euclid,
        distance_harmonic,
        distance_logchol,
        distance_logeuclid,
        distance_riemann,
        distance_wasserstein,
    ],
    ids=[
        "chol",
        "euclid",
        "harmonic",
        "logchol",
        "logeuclid",
        "riemann",
        "wasserstein",
    ],
)
def test_torch_distances_match_numpy_with_broadcast(fn):
    rng = np.random.RandomState(42)
    A_np = _make_spd((2, 1), 3, rng)
    B_np = _make_spd((1, 3), 3, rng)

    result = fn(_to_torch(A_np), _to_torch(B_np))

    expected = np.empty((2, 3))
    for i in range(2):
        for j in range(3):
            expected[i, j] = fn(A_np[i, 0], B_np[0, j])

    assert isinstance(result, torch.Tensor)
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        expected,
        **STRICT_TOL,
    )


@pytest.mark.parametrize(
    "fn",
    [
        geodesic_chol,
        geodesic_logchol,
        geodesic_logeuclid,
        geodesic_riemann,
        geodesic_wasserstein,
    ],
    ids=["chol", "logchol", "logeuclid", "riemann", "wasserstein"],
)
def test_torch_geodesics_match_numpy_with_broadcast(fn):
    rng = np.random.RandomState(7)
    A_np = _make_spd((2, 1), 3, rng)
    B_np = _make_spd((1, 3), 3, rng)
    alpha = 0.25

    result = fn(_to_torch(A_np), _to_torch(B_np), alpha=alpha)

    expected = np.empty((2, 3, 3, 3))
    for i in range(2):
        for j in range(3):
            expected[i, j] = fn(A_np[i, 0], B_np[0, j], alpha=alpha)

    assert isinstance(result, torch.Tensor)
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        expected,
        **STRICT_TOL,
    )


@pytest.mark.parametrize(
    ("fn", "tol"),
    [
        (mean_chol, STRICT_TOL),
        (mean_harmonic, STRICT_TOL),
        (mean_logchol, STRICT_TOL),
        (mean_logeuclid, STRICT_TOL),
        (mean_riemann, ITER_TOL),
        (mean_wasserstein, ITER_TOL),
    ],
    ids=["chol", "harmonic", "logchol", "logeuclid", "riemann",
         "wasserstein"],
)
def test_torch_means_match_numpy(fn, tol):
    rng = np.random.RandomState(11)
    X_np = _make_spd((4,), 3, rng)
    weights_np = np.array([0.1, 0.2, 0.3, 0.4])

    result = fn(_to_torch(X_np), sample_weight=_to_torch(weights_np))
    expected = fn(X_np, sample_weight=weights_np)

    assert isinstance(result, torch.Tensor)
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        expected,
        **tol,
    )


def test_riemann_maps_roundtrip_with_broadcast():
    rng = np.random.RandomState(21)
    Cref_np = _make_spd((2, 1), 3, rng)
    X_np = _make_spd((1, 3), 3, rng)

    Cref_t = _to_torch(Cref_np)
    X_t = _to_torch(X_np)

    tangent = log_map_riemann(X_t, Cref_t, C12=True)
    recovered = exp_map_riemann(tangent, Cref_t, Cm12=True)

    assert isinstance(tangent, torch.Tensor)
    assert isinstance(recovered, torch.Tensor)
    np.testing.assert_allclose(
        recovered.detach().cpu().numpy(),
        np.broadcast_to(X_np, recovered.shape),
        **STRICT_TOL,
    )


def test_logchol_maps_roundtrip_with_broadcast():
    rng = np.random.RandomState(23)
    Cref_np = _make_spd((2, 1), 3, rng)
    X_np = _make_spd((1, 3), 3, rng)

    Cref_t = _to_torch(Cref_np)
    X_t = _to_torch(X_np)

    tangent = log_map_logchol(X_t, Cref_t)
    recovered = exp_map_logchol(tangent, Cref_t)

    assert isinstance(tangent, torch.Tensor)
    assert isinstance(recovered, torch.Tensor)
    np.testing.assert_allclose(
        recovered.detach().cpu().numpy(),
        np.broadcast_to(X_np, recovered.shape),
        **STRICT_TOL,
    )


@pytest.mark.parametrize(
    "metric",
    ["chol", "euclid", "harmonic", "logchol", "logeuclid", "riemann",
     "wasserstein"],
)
def test_torch_distance_wrapper_matches_numpy(metric):
    rng = np.random.RandomState(31)
    X_np = _make_spd((4,), 3, rng)
    B_np = _make_spd((), 3, rng)

    result = distance(_to_torch(X_np), _to_torch(B_np), metric=metric)
    expected = distance(X_np, B_np, metric=metric)

    assert isinstance(result, torch.Tensor)
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        expected,
        **STRICT_TOL,
    )


@pytest.mark.parametrize(
    "metric",
    ["chol", "euclid", "harmonic", "logchol", "logeuclid", "riemann",
     "wasserstein"],
)
def test_torch_pairwise_distance_matches_numpy(metric):
    rng = np.random.RandomState(37)
    X_np = _make_spd((4,), 3, rng)
    Y_np = _make_spd((3,), 3, rng)

    result = pairwise_distance(_to_torch(X_np), _to_torch(Y_np), metric=metric)
    expected = pairwise_distance(X_np, Y_np, metric=metric)

    assert isinstance(result, torch.Tensor)
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        expected,
        **STRICT_TOL,
    )


def test_torch_upper_unupper_roundtrip():
    rng = np.random.RandomState(41)
    X_np = _make_spd((2, 4), 3, rng)

    T = upper(_to_torch(X_np))
    recovered = unupper(T)

    assert isinstance(T, torch.Tensor)
    assert isinstance(recovered, torch.Tensor)
    np.testing.assert_allclose(
        recovered.detach().cpu().numpy(),
        X_np,
        **STRICT_TOL,
    )


@pytest.mark.parametrize("metric", ["euclid", "logchol", "riemann",
                                    "wasserstein"])
def test_torch_tangent_space_roundtrip(metric):
    rng = np.random.RandomState(43)
    X_np = _make_spd((4,), 3, rng)
    Cref_np = _make_spd((), 3, rng)

    tangent = tangent_space(_to_torch(X_np), _to_torch(Cref_np), metric=metric)
    recovered = untangent_space(tangent, _to_torch(Cref_np), metric=metric)

    assert isinstance(tangent, torch.Tensor)
    assert isinstance(recovered, torch.Tensor)
    np.testing.assert_allclose(
        recovered.detach().cpu().numpy(),
        X_np,
        **ITER_TOL if metric == "wasserstein" else STRICT_TOL,
    )


def test_torch_backend_autograd_smoke():
    rng = np.random.RandomState(99)
    A = (
        _to_torch(_make_spd((), 3, rng)).clone().detach().requires_grad_(True)
    )
    B = (
        _to_torch(_make_spd((), 3, rng)).clone().detach().requires_grad_(True)
    )
    X = _to_torch(_make_spd((4,), 3, rng))
    X = X.clone().detach().requires_grad_(True)

    tangent = log_map_riemann(B.unsqueeze(0), A, C12=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss = distance_riemann(A, B)
        loss = loss + geodesic_logchol(A, B, alpha=0.25).sum()
        loss = loss + mean_logeuclid(X).sum()
        loss = loss + mean_riemann(X, maxiter=5).sum()
        loss = loss + mean_wasserstein(X, maxiter=5).sum()
        loss = loss + exp_map_riemann(tangent, A, Cm12=True).sum()
    loss.backward()

    assert A.grad is not None
    assert B.grad is not None
    assert X.grad is not None
    assert torch.isfinite(A.grad).all()
    assert torch.isfinite(B.grad).all()
    assert torch.isfinite(X.grad).all()


def test_torch_matrix_operator_rejects_non_finite_input():
    bad = torch.tensor([[float("nan"), 0.0], [0.0, 1.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="Matrices must be positive definite"):
        sqrtm(bad)
