import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.utils.base import sqrtm  # noqa: E402
from pyriemann.utils.distance import (  # noqa: E402
    distance,
    distance_chol,
    distance_euclid,
    distance_harmonic,
    distance_logchol,
    distance_logeuclid,
    distance_riemann,
    distance_wasserstein,
    pairwise_distance,
)
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
    exp_map_logchol,
    exp_map_riemann,
    log_map_logchol,
    log_map_riemann,
    tangent_space,
    untangent_space,
    upper,
    unupper,
)


STRICT_TOL = dict(atol=1e-7, rtol=1e-6)
ITER_TOL = dict(atol=1e-6, rtol=1e-5)
DISTANCE_METRICS = [
    "chol",
    "euclid",
    "harmonic",
    "logchol",
    "logeuclid",
    "riemann",
    "wasserstein",
]
TANGENT_METRICS = ["euclid", "logchol", "riemann", "wasserstein"]


def _make_spd(batch_shape, n, rng):
    mats = rng.standard_normal((*batch_shape, n, n))
    return mats @ np.swapaxes(mats, -1, -2) + 0.25 * np.eye(n)


def _to_torch(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to(torch.float64)


def _torchify(x):
    return _to_torch(x) if isinstance(x, np.ndarray) else x


def _assert_same_result(fn, *args, tol=STRICT_TOL, **kwargs):
    torch_args = tuple(_torchify(arg) for arg in args)
    torch_kwargs = {key: _torchify(value) for key, value in kwargs.items()}
    result = fn(*torch_args, **torch_kwargs)
    expected = fn(*args, **kwargs)

    assert isinstance(result, torch.Tensor)
    np.testing.assert_allclose(
        result.detach().cpu().numpy(),
        expected,
        **tol,
    )


def _assert_same_error(fn, *args, match=None, **kwargs):
    with pytest.raises(ValueError, match=match):
        fn(*args, **kwargs)

    torch_args = tuple(_torchify(arg) for arg in args)
    torch_kwargs = {key: _torchify(value) for key, value in kwargs.items()}
    with pytest.raises(ValueError, match=match):
        fn(*torch_args, **torch_kwargs)


def _broadcast_pair(seed):
    rng = np.random.RandomState(seed)
    return _make_spd((2, 1), 3, rng), _make_spd((1, 3), 3, rng)


def _weighted_set(seed):
    rng = np.random.RandomState(seed)
    X = _make_spd((4,), 3, rng)
    weights = np.array([0.1, 0.2, 0.3, 0.4])
    return X, weights


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
    ids=lambda fn: fn.__name__,
)
def test_torch_distances_match_numpy(fn):
    _assert_same_result(fn, *_broadcast_pair(seed=42))


@pytest.mark.parametrize(
    "fn",
    [
        geodesic_chol,
        geodesic_logchol,
        geodesic_logeuclid,
        geodesic_riemann,
        geodesic_wasserstein,
    ],
    ids=lambda fn: fn.__name__,
)
def test_torch_geodesics_match_numpy(fn):
    _assert_same_result(
        fn,
        *_broadcast_pair(seed=7),
        alpha=0.25,
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
    ids=[
        "mean_chol",
        "mean_harmonic",
        "mean_logchol",
        "mean_logeuclid",
        "mean_riemann",
        "mean_wasserstein",
    ],
)
def test_torch_means_match_numpy(fn, tol):
    X, weights = _weighted_set(seed=11)
    _assert_same_result(fn, X, sample_weight=weights, tol=tol)


@pytest.mark.parametrize(
    ("fn", "args", "kwargs"),
    [
        (log_map_logchol, _broadcast_pair(seed=21), {}),
        (log_map_riemann, _broadcast_pair(seed=22), {"C12": True}),
        (
            exp_map_logchol,
            (
                log_map_logchol(*_broadcast_pair(seed=23)),
                _broadcast_pair(seed=23)[1],
            ),
            {},
        ),
        (
            exp_map_riemann,
            (
                log_map_riemann(*_broadcast_pair(seed=24), C12=True),
                _broadcast_pair(seed=24)[1],
            ),
            {"Cm12": True},
        ),
    ],
    ids=[
        "log_map_logchol",
        "log_map_riemann",
        "exp_map_logchol",
        "exp_map_riemann",
    ],
)
def test_torch_maps_match_numpy(fn, args, kwargs):
    _assert_same_result(fn, *args, **kwargs)


@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_torch_distance_wrapper_matches_numpy(metric):
    rng = np.random.RandomState(31)
    X = _make_spd((4,), 3, rng)
    Cref = _make_spd((), 3, rng)
    _assert_same_result(distance, X, Cref, metric=metric)


@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_torch_pairwise_distance_matches_numpy(metric):
    rng = np.random.RandomState(37)
    X = _make_spd((4,), 3, rng)
    Y = _make_spd((3,), 3, rng)
    _assert_same_result(pairwise_distance, X, Y, metric=metric)


@pytest.mark.parametrize(
    ("fn", "arg_factory"),
    [
        (
            upper,
            lambda: (_make_spd((2, 4), 3, np.random.RandomState(41)),),
        ),
        (
            unupper,
            lambda: (
                upper(_make_spd((2, 4), 3, np.random.RandomState(42))),
            ),
        ),
    ],
    ids=["upper", "unupper"],
)
def test_torch_tangent_helpers_match_numpy(fn, arg_factory):
    _assert_same_result(fn, *arg_factory())


@pytest.mark.parametrize("metric", TANGENT_METRICS)
def test_torch_tangent_space_matches_numpy(metric):
    rng = np.random.RandomState(43)
    X = _make_spd((4,), 3, rng)
    Cref = _make_spd((), 3, rng)
    _assert_same_result(tangent_space, X, Cref, metric=metric)


@pytest.mark.parametrize("metric", TANGENT_METRICS)
def test_torch_untangent_space_matches_numpy(metric):
    rng = np.random.RandomState(44)
    X = _make_spd((4,), 3, rng)
    Cref = _make_spd((), 3, rng)
    T = tangent_space(X, Cref, metric=metric)
    tol = ITER_TOL if metric == "wasserstein" else STRICT_TOL
    _assert_same_result(untangent_space, T, Cref, metric=metric, tol=tol)


def test_torch_backend_autograd_smoke():
    rng = np.random.RandomState(99)
    A = _to_torch(_make_spd((), 3, rng)).clone().detach().requires_grad_(True)
    B = _to_torch(_make_spd((), 3, rng)).clone().detach().requires_grad_(True)
    X = _to_torch(_make_spd((4,), 3, rng)).clone().detach()
    X = X.requires_grad_(True)

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


def test_torch_matrix_operator_matches_numpy_error():
    bad = np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float64)
    _assert_same_error(
        sqrtm,
        bad,
        match="Matrices must be positive definite",
    )
