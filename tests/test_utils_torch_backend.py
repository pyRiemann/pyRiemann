import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pyriemann.utils.ajd import ajd, rjd, uwedge  # noqa: E402
from pyriemann.utils.base import (  # noqa: E402
    ddexpm,
    ddlogm,
    nearest_sym_pos_def,
    sqrtm,
)
from pyriemann.utils.covariance import (  # noqa: E402
    block_covariances,
    coherence,
    covariance_mest,
    covariance_sch,
    covariance_scm,
    covariances,
    covariances_EP,
    covariances_X,
    cospectrum,
    cross_spectrum,
    eegtocov,
    get_nondiag_weight,
    normalize,
)
from pyriemann.utils.distance import (  # noqa: E402
    distance,
    distance_chol,
    distance_euclid,
    distance_harmonic,
    distance_kullback,
    distance_kullback_right,
    distance_kullback_sym,
    distance_logchol,
    distance_logdet,
    distance_logeuclid,
    distance_mahalanobis,
    distance_poweuclid,
    distance_riemann,
    distance_thompson,
    distance_wasserstein,
    pairwise_distance,
)
from pyriemann.utils.geodesic import (  # noqa: E402
    geodesic_chol,
    geodesic_euclid,
    geodesic_logchol,
    geodesic_logeuclid,
    geodesic_riemann,
    geodesic_thompson,
    geodesic_wasserstein,
)
from pyriemann.utils.kernel import (  # noqa: E402
    kernel,
    kernel_euclid,
    kernel_logeuclid,
    kernel_riemann,
)
from pyriemann.utils.median import median_euclid, median_riemann  # noqa: E402
from pyriemann.utils.mean import (  # noqa: E402
    mean_ale,
    mean_alm,
    mean_chol,
    mean_harmonic,
    mean_logchol,
    mean_logdet,
    mean_logeuclid,
    mean_power,
    mean_poweuclid,
    mean_riemann,
    mean_thompson,
    mean_wasserstein,
    maskedmean_riemann,
    nanmean_riemann,
)
from pyriemann.utils.tangentspace import (  # noqa: E402
    exp_map_euclid,
    exp_map_logchol,
    exp_map_logeuclid,
    exp_map_riemann,
    log_map_euclid,
    log_map_logchol,
    log_map_logeuclid,
    log_map_riemann,
    tangent_space,
    transport_euclid,
    transport_logchol,
    transport_logeuclid,
    transport_riemann,
    untangent_space,
    upper,
    unupper,
)
from pyriemann.utils.test import (  # noqa: E402
    is_hankel,
    is_pos_def,
    is_real_type,
    is_square,
    is_sym,
)


STRICT_TOL = dict(atol=1e-7, rtol=1e-6)
ITER_TOL = dict(atol=1e-6, rtol=1e-5)
DISTANCE_METRICS = [
    "chol",
    "euclid",
    "harmonic",
    "kullback",
    "kullback_right",
    "kullback_sym",
    "logchol",
    "logdet",
    "logeuclid",
    "riemann",
    "thompson",
    "wasserstein",
]
TANGENT_METRICS = ["euclid", "logchol", "riemann", "wasserstein"]
KERNEL_METRICS = ["euclid", "logeuclid", "riemann"]


def _make_spd(batch_shape, n, rng):
    mats = rng.standard_normal((*batch_shape, n, n))
    return mats @ np.swapaxes(mats, -1, -2) + 0.25 * np.eye(n)


def _make_sym(batch_shape, n, rng):
    mats = rng.standard_normal((*batch_shape, n, n))
    return 0.5 * (mats + np.swapaxes(mats, -1, -2))


def _to_torch(x):
    dtype = torch.complex128 if np.iscomplexobj(x) else torch.float64
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype)


def _torchify(x):
    if isinstance(x, np.ndarray):
        return _to_torch(x)
    if isinstance(x, list):
        return [_torchify(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_torchify(v) for v in x)
    if isinstance(x, dict):
        return {k: _torchify(v) for k, v in x.items()}
    return x


def _assert_same_result(fn, *args, tol=STRICT_TOL, **kwargs):
    torch_args = tuple(_torchify(arg) for arg in args)
    torch_kwargs = {key: _torchify(value) for key, value in kwargs.items()}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = fn(*torch_args, **torch_kwargs)
        expected = fn(*args, **kwargs)

    _assert_same_value(result, expected, tol=tol)


def _assert_same_value(result, expected, *, tol):
    if expected is None:
        assert result is None
    elif isinstance(expected, tuple):
        assert isinstance(result, tuple)
        assert len(result) == len(expected)
        for result_i, expected_i in zip(result, expected):
            _assert_same_value(result_i, expected_i, tol=tol)
    elif isinstance(expected, list):
        assert isinstance(result, list)
        assert len(result) == len(expected)
        for result_i, expected_i in zip(result, expected):
            _assert_same_value(result_i, expected_i, tol=tol)
    elif isinstance(expected, (bool, np.bool_)):
        assert result == expected
    elif np.isscalar(expected):
        if isinstance(result, torch.Tensor):
            result = result.detach().cpu().numpy()
        np.testing.assert_allclose(result, expected, **tol)
    else:
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


def _weighted_set(seed, n_matrices=4):
    rng = np.random.RandomState(seed)
    X = _make_spd((n_matrices,), 3, rng)
    weights = np.arange(1, n_matrices + 1, dtype=np.float64)
    return X, weights / weights.sum()


def _transport_args(seed):
    rng = np.random.RandomState(seed)
    return _make_sym((), 3, rng), _make_spd((), 3, rng), _make_spd((), 3, rng)


def _vector_args(seed):
    rng = np.random.RandomState(seed)
    return (
        rng.standard_normal((3, 5)),
        _make_spd((), 3, rng),
        rng.standard_normal((3, 1)),
    )


def _nearest_spd_args(seed):
    rng = np.random.RandomState(seed)
    X = _make_sym((3,), 3, rng)
    return (X - 2 * np.eye(3),)


def _ajd_args(seed):
    rng = np.random.RandomState(seed)
    return (_make_spd((6,), 3, rng),)


def _masked_mean_args(seed):
    rng = np.random.RandomState(seed)
    X = _make_spd((4,), 3, rng)
    masks = [
        np.eye(3)[:, [0, 1]],
        np.eye(3)[:, [0, 2]],
        np.eye(3)[:, [1, 2]],
        np.eye(3),
    ]
    return X, masks


def _nan_mean_args(seed):
    X, _ = _masked_mean_args(seed)
    X = X.copy()
    X[0, 2, :] = np.nan
    X[0, :, 2] = np.nan
    X[1, 0, :] = np.nan
    X[1, :, 0] = np.nan
    return (X,)


def _signal(seed, n_channels=3, n_times=64):
    rng = np.random.RandomState(seed)
    return rng.standard_normal((n_channels, n_times))


def _signal_batch(seed, n_matrices=3, n_channels=3, n_times=64):
    rng = np.random.RandomState(seed)
    return rng.standard_normal((n_matrices, n_channels, n_times))


def _complex_signal_batch(seed, n_matrices=3, n_channels=3, n_times=64):
    rng = np.random.RandomState(seed)
    real = rng.standard_normal((n_matrices, n_channels, n_times))
    imag = rng.standard_normal((n_matrices, n_channels, n_times))
    return real + 1j * imag


def _ep_args(seed):
    X = _signal_batch(seed)
    P = _signal(seed + 1, n_channels=2, n_times=X.shape[-1])
    return X, P


def _kernel_pair(seed, n_matrices_x=4, n_matrices_y=3, n=3):
    rng = np.random.RandomState(seed)
    return (
        _make_spd((n_matrices_x,), n, rng),
        _make_spd((n_matrices_y,), n, rng),
    )


def _eeg_args(seed):
    rng = np.random.RandomState(seed)
    return (rng.standard_normal((256, 3)),)


def _square_args(seed):
    rng = np.random.RandomState(seed)
    X = _make_spd((4,), 3, rng)
    Y = _make_sym((), 3, rng)
    return X, Y


DISTANCE_CASES = [
    (
        "distance_chol",
        distance_chol,
        lambda: _broadcast_pair(42),
        {},
        STRICT_TOL,
    ),
    (
        "distance_euclid",
        distance_euclid,
        lambda: _broadcast_pair(43),
        {},
        STRICT_TOL,
    ),
    (
        "distance_harmonic",
        distance_harmonic,
        lambda: _broadcast_pair(44),
        {},
        STRICT_TOL,
    ),
    (
        "distance_kullback",
        distance_kullback,
        lambda: _broadcast_pair(45),
        {},
        STRICT_TOL,
    ),
    (
        "distance_kullback_right",
        distance_kullback_right,
        lambda: _broadcast_pair(46),
        {},
        STRICT_TOL,
    ),
    (
        "distance_kullback_sym",
        distance_kullback_sym,
        lambda: _broadcast_pair(47),
        {},
        STRICT_TOL,
    ),
    (
        "distance_logchol",
        distance_logchol,
        lambda: _broadcast_pair(48),
        {},
        STRICT_TOL,
    ),
    (
        "distance_logdet",
        distance_logdet,
        lambda: _broadcast_pair(49),
        {},
        STRICT_TOL,
    ),
    (
        "distance_logeuclid",
        distance_logeuclid,
        lambda: _broadcast_pair(50),
        {},
        STRICT_TOL,
    ),
    (
        "distance_poweuclid",
        distance_poweuclid,
        lambda: _broadcast_pair(51),
        {"p": 0.5},
        STRICT_TOL,
    ),
    (
        "distance_riemann",
        distance_riemann,
        lambda: _broadcast_pair(52),
        {},
        STRICT_TOL,
    ),
    (
        "distance_thompson",
        distance_thompson,
        lambda: _broadcast_pair(53),
        {},
        STRICT_TOL,
    ),
    (
        "distance_wasserstein",
        distance_wasserstein,
        lambda: _broadcast_pair(54),
        {},
        STRICT_TOL,
    ),
    (
        "distance_mahalanobis",
        distance_mahalanobis,
        lambda: _vector_args(55),
        {},
        STRICT_TOL,
    ),
]

GEODESIC_CASES = [
    (
        "geodesic_chol",
        geodesic_chol,
        lambda: _broadcast_pair(60),
        {"alpha": 0.25},
    ),
    (
        "geodesic_euclid",
        geodesic_euclid,
        lambda: _broadcast_pair(61),
        {"alpha": 0.25},
    ),
    (
        "geodesic_logchol",
        geodesic_logchol,
        lambda: _broadcast_pair(62),
        {"alpha": 0.25},
    ),
    (
        "geodesic_logeuclid",
        geodesic_logeuclid,
        lambda: _broadcast_pair(63),
        {"alpha": 0.25},
    ),
    (
        "geodesic_riemann",
        geodesic_riemann,
        lambda: _broadcast_pair(64),
        {"alpha": 0.25},
    ),
    (
        "geodesic_thompson",
        geodesic_thompson,
        lambda: _broadcast_pair(65),
        {"alpha": 0.25},
    ),
    (
        "geodesic_wasserstein",
        geodesic_wasserstein,
        lambda: _broadcast_pair(66),
        {"alpha": 0.25},
    ),
]

MEAN_CASES = [
    ("mean_ale", mean_ale, lambda: _weighted_set(70), {}, ITER_TOL),
    (
        "mean_alm",
        mean_alm,
        lambda: _weighted_set(71, n_matrices=3),
        {},
        ITER_TOL,
    ),
    ("mean_chol", mean_chol, lambda: _weighted_set(72), {}, STRICT_TOL),
    (
        "mean_harmonic",
        mean_harmonic,
        lambda: _weighted_set(73),
        {},
        STRICT_TOL,
    ),
    (
        "mean_logchol",
        mean_logchol,
        lambda: _weighted_set(74),
        {},
        STRICT_TOL,
    ),
    ("mean_logdet", mean_logdet, lambda: _weighted_set(75), {}, ITER_TOL),
    (
        "mean_logeuclid",
        mean_logeuclid,
        lambda: _weighted_set(76),
        {},
        STRICT_TOL,
    ),
    (
        "mean_power",
        mean_power,
        lambda: _weighted_set(77),
        {"p": 0.5},
        ITER_TOL,
    ),
    (
        "mean_poweuclid",
        mean_poweuclid,
        lambda: _weighted_set(78),
        {"p": 0.5},
        STRICT_TOL,
    ),
    ("mean_riemann", mean_riemann, lambda: _weighted_set(79), {}, ITER_TOL),
    (
        "mean_thompson",
        mean_thompson,
        lambda: _weighted_set(80),
        {},
        ITER_TOL,
    ),
    (
        "mean_wasserstein",
        mean_wasserstein,
        lambda: _weighted_set(81),
        {},
        ITER_TOL,
    ),
]

MAP_CASES = [
    (
        "exp_map_euclid",
        exp_map_euclid,
        lambda: _transport_args(90)[:2],
        {},
        STRICT_TOL,
    ),
    (
        "log_map_euclid",
        log_map_euclid,
        lambda: _transport_args(91)[:2],
        {},
        STRICT_TOL,
    ),
    (
        "log_map_logchol",
        log_map_logchol,
        lambda: _broadcast_pair(92),
        {},
        STRICT_TOL,
    ),
    (
        "log_map_logeuclid",
        log_map_logeuclid,
        lambda: _broadcast_pair(93),
        {},
        STRICT_TOL,
    ),
    (
        "log_map_riemann",
        log_map_riemann,
        lambda: _broadcast_pair(94),
        {"C12": True},
        STRICT_TOL,
    ),
    (
        "exp_map_logchol",
        exp_map_logchol,
        lambda: (
            log_map_logchol(*_broadcast_pair(95)),
            _broadcast_pair(95)[1],
        ),
        {},
        STRICT_TOL,
    ),
    (
        "exp_map_logeuclid",
        exp_map_logeuclid,
        lambda: (
            log_map_logeuclid(*_broadcast_pair(96)),
            _broadcast_pair(96)[1],
        ),
        {},
        STRICT_TOL,
    ),
    (
        "exp_map_riemann",
        exp_map_riemann,
        lambda: (
            log_map_riemann(*_broadcast_pair(97), C12=True),
            _broadcast_pair(97)[1],
        ),
        {"Cm12": True},
        STRICT_TOL,
    ),
]

TRANSPORT_CASES = [
    (
        "transport_euclid",
        transport_euclid,
        lambda: _transport_args(100),
        {},
        STRICT_TOL,
    ),
    (
        "transport_logchol",
        transport_logchol,
        lambda: _transport_args(101),
        {},
        STRICT_TOL,
    ),
    (
        "transport_logeuclid",
        transport_logeuclid,
        lambda: _transport_args(102),
        {},
        STRICT_TOL,
    ),
    (
        "transport_riemann",
        transport_riemann,
        lambda: _transport_args(103),
        {},
        STRICT_TOL,
    ),
]

UTILITY_CASES = [
    (
        "nearest_sym_pos_def",
        nearest_sym_pos_def,
        lambda: _nearest_spd_args(110),
        {},
    ),
    ("ddexpm", ddexpm, lambda: _transport_args(111)[:2], {}),
    ("ddlogm", ddlogm, lambda: _transport_args(112)[:2], {}),
]

AJD_CASES = [
    ("rjd", rjd, lambda: _ajd_args(150), {}, ITER_TOL),
    ("uwedge", uwedge, lambda: _ajd_args(151), {}, ITER_TOL),
    (
        "ajd_rjd",
        ajd,
        lambda: _ajd_args(152),
        {"method": "rjd"},
        ITER_TOL,
    ),
    (
        "ajd_uwedge",
        ajd,
        lambda: _ajd_args(153),
        {"method": "uwedge"},
        ITER_TOL,
    ),
]

EXTRA_MEAN_CASES = [
    (
        "maskedmean_riemann",
        maskedmean_riemann,
        lambda: _masked_mean_args(160),
        {"maxiter": 10},
        ITER_TOL,
    ),
    (
        "nanmean_riemann",
        nanmean_riemann,
        lambda: _nan_mean_args(161),
        {"maxiter": 4},
        ITER_TOL,
    ),
]

MEDIAN_CASES = [
    (
        "median_euclid",
        median_euclid,
        lambda: (_weighted_set(170)[0],),
        {"weights": _weighted_set(170)[1], "maxiter": 10},
        ITER_TOL,
    ),
    (
        "median_riemann",
        median_riemann,
        lambda: (_weighted_set(171)[0],),
        {"weights": _weighted_set(171)[1], "maxiter": 10},
        ITER_TOL,
    ),
]

COVARIANCE_CASES = [
    (
        "covariance_scm",
        covariance_scm,
        lambda: (_signal(180),),
        {},
        STRICT_TOL,
    ),
    (
        "covariance_sch",
        covariance_sch,
        lambda: (_signal(181),),
        {},
        STRICT_TOL,
    ),
    (
        "covariance_mest_hub",
        covariance_mest,
        lambda: (_signal(182), "hub"),
        {"n_iter_max": 5},
        ITER_TOL,
    ),
    (
        "covariance_mest_stu",
        covariance_mest,
        lambda: (_signal(183), "stu"),
        {"n_iter_max": 5},
        ITER_TOL,
    ),
    (
        "covariance_mest_tyl",
        covariance_mest,
        lambda: (_signal(184), "tyl"),
        {"n_iter_max": 5},
        ITER_TOL,
    ),
    (
        "covariances_EP",
        covariances_EP,
        lambda: _ep_args(185),
        {"estimator": "scm"},
        STRICT_TOL,
    ),
    (
        "covariances_X",
        covariances_X,
        lambda: (_signal_batch(186),),
        {"estimator": "scm", "alpha": 0.5},
        STRICT_TOL,
    ),
    (
        "block_covariances",
        block_covariances,
        lambda: (_signal_batch(187), [1, 2]),
        {"estimator": "scm"},
        STRICT_TOL,
    ),
    ("eegtocov", eegtocov, lambda: _eeg_args(188), {"window": 32}, STRICT_TOL),
    (
        "cross_spectrum",
        cross_spectrum,
        lambda: (_signal(189, n_times=256),),
        {"window": 64, "fs": 128},
        STRICT_TOL,
    ),
    (
        "cospectrum",
        cospectrum,
        lambda: (_signal(190, n_times=256),),
        {"window": 64, "fs": 128},
        STRICT_TOL,
    ),
    (
        "coherence",
        coherence,
        lambda: (_signal(191, n_times=256),),
        {"window": 64, "fs": 128},
        STRICT_TOL,
    ),
    (
        "normalize_trace",
        normalize,
        lambda: (_make_spd((4,), 3, np.random.RandomState(192)), "trace"),
        {},
        STRICT_TOL,
    ),
    (
        "get_nondiag_weight",
        get_nondiag_weight,
        lambda: (_make_sym((4,), 3, np.random.RandomState(193)),),
        {},
        STRICT_TOL,
    ),
]

KERNEL_CASES = [
    (
        "kernel_euclid",
        kernel_euclid,
        lambda: _kernel_pair(195),
        {},
        STRICT_TOL,
    ),
    (
        "kernel_logeuclid",
        kernel_logeuclid,
        lambda: _kernel_pair(196),
        {},
        STRICT_TOL,
    ),
    (
        "kernel_riemann",
        kernel_riemann,
        lambda: _kernel_pair(197),
        {},
        ITER_TOL,
    ),
]

PREDICATE_CASES = [
    (
        "is_square",
        is_square,
        lambda: (_make_spd((4,), 3, np.random.RandomState(200)),),
    ),
    (
        "is_sym",
        is_sym,
        lambda: (_make_spd((4,), 3, np.random.RandomState(201)),),
    ),
    (
        "is_hankel",
        is_hankel,
        lambda: (np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]),),
    ),
    ("is_real_type", is_real_type, lambda: (_signal(202),)),
    (
        "is_pos_def",
        is_pos_def,
        lambda: (_make_spd((4,), 3, np.random.RandomState(203)),),
    ),
]


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    DISTANCE_CASES,
    ids=[case[0] for case in DISTANCE_CASES],
)
def test_torch_distances_match_numpy(_name, fn, arg_factory, kwargs, tol):
    _assert_same_result(fn, *arg_factory(), tol=tol, **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs"),
    GEODESIC_CASES,
    ids=[case[0] for case in GEODESIC_CASES],
)
def test_torch_geodesics_match_numpy(_name, fn, arg_factory, kwargs):
    _assert_same_result(fn, *arg_factory(), **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    MEAN_CASES,
    ids=[case[0] for case in MEAN_CASES],
)
def test_torch_means_match_numpy(_name, fn, arg_factory, kwargs, tol):
    X, weights = arg_factory()
    _assert_same_result(fn, X, sample_weight=weights, tol=tol, **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    MAP_CASES,
    ids=[case[0] for case in MAP_CASES],
)
def test_torch_maps_match_numpy(_name, fn, arg_factory, kwargs, tol):
    _assert_same_result(fn, *arg_factory(), tol=tol, **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    TRANSPORT_CASES,
    ids=[case[0] for case in TRANSPORT_CASES],
)
def test_torch_transports_match_numpy(_name, fn, arg_factory, kwargs, tol):
    _assert_same_result(fn, *arg_factory(), tol=tol, **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs"),
    UTILITY_CASES,
    ids=[case[0] for case in UTILITY_CASES],
)
def test_torch_backend_utilities_match_numpy(_name, fn, arg_factory, kwargs):
    _assert_same_result(fn, *arg_factory(), **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    AJD_CASES,
    ids=[case[0] for case in AJD_CASES],
)
def test_torch_ajd_matches_numpy(_name, fn, arg_factory, kwargs, tol):
    _assert_same_result(fn, *arg_factory(), tol=tol, **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    EXTRA_MEAN_CASES,
    ids=[case[0] for case in EXTRA_MEAN_CASES],
)
def test_torch_extra_means_match_numpy(_name, fn, arg_factory, kwargs, tol):
    _assert_same_result(fn, *arg_factory(), tol=tol, **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    MEDIAN_CASES,
    ids=[case[0] for case in MEDIAN_CASES],
)
def test_torch_medians_match_numpy(_name, fn, arg_factory, kwargs, tol):
    _assert_same_result(fn, *arg_factory(), tol=tol, **kwargs)


@pytest.mark.parametrize(
    "estimator",
    ["corr", "cov", "hub", "lwf", "mcd", "oas", "sch", "scm", "stu", "tyl"],
)
def test_torch_covariances_match_numpy(estimator):
    kwargs = {"n_iter_max": 5} if estimator in {"hub", "stu", "tyl"} else {}
    if estimator == "mcd":
        kwargs["random_state"] = 0
    _assert_same_result(
        covariances,
        _signal_batch(194, n_times=96),
        estimator=estimator,
        tol=ITER_TOL if estimator in {"hub", "stu", "tyl"} else STRICT_TOL,
        **kwargs,
    )


@pytest.mark.parametrize("estimator", ["lwf", "mcd", "oas", "sch", "scm"])
def test_torch_complex_covariances_match_numpy(estimator):
    kwargs = {"random_state": 0} if estimator == "mcd" else {}
    _assert_same_result(
        covariances,
        _complex_signal_batch(199, n_times=72),
        estimator=estimator,
        tol=STRICT_TOL,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    COVARIANCE_CASES,
    ids=[case[0] for case in COVARIANCE_CASES],
)
def test_torch_covariance_helpers_match_numpy(
    _name,
    fn,
    arg_factory,
    kwargs,
    tol,
):
    _assert_same_result(fn, *arg_factory(), tol=tol, **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory", "kwargs", "tol"),
    KERNEL_CASES,
    ids=[case[0] for case in KERNEL_CASES],
)
def test_torch_kernel_helpers_match_numpy(
    _name,
    fn,
    arg_factory,
    kwargs,
    tol,
):
    _assert_same_result(fn, *arg_factory(), tol=tol, **kwargs)


@pytest.mark.parametrize(
    ("_name", "fn", "arg_factory"),
    PREDICATE_CASES,
    ids=[case[0] for case in PREDICATE_CASES],
)
def test_torch_predicates_match_numpy(_name, fn, arg_factory):
    _assert_same_result(fn, *arg_factory())


@pytest.mark.parametrize("metric", KERNEL_METRICS)
def test_torch_kernel_wrapper_matches_numpy(metric):
    X, Y = _kernel_pair(198)
    tol = ITER_TOL if metric == "riemann" else STRICT_TOL
    _assert_same_result(kernel, X, Y, metric=metric, tol=tol)


@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_torch_distance_wrapper_matches_numpy(metric):
    rng = np.random.RandomState(120)
    X = _make_spd((4,), 3, rng)
    Cref = _make_spd((), 3, rng)
    _assert_same_result(distance, X, Cref, metric=metric)


@pytest.mark.parametrize("metric", DISTANCE_METRICS)
def test_torch_pairwise_distance_matches_numpy(metric):
    rng = np.random.RandomState(121)
    X = _make_spd((4,), 3, rng)
    Y = _make_spd((3,), 3, rng)
    _assert_same_result(pairwise_distance, X, Y, metric=metric)


@pytest.mark.parametrize(
    ("fn", "arg_factory"),
    [
        (
            upper,
            lambda: (_make_spd((2, 4), 3, np.random.RandomState(130)),),
        ),
        (
            unupper,
            lambda: (
                upper(_make_spd((2, 4), 3, np.random.RandomState(131))),
            ),
        ),
    ],
    ids=["upper", "unupper"],
)
def test_torch_tangent_helpers_match_numpy(fn, arg_factory):
    _assert_same_result(fn, *arg_factory())


@pytest.mark.parametrize("metric", TANGENT_METRICS)
def test_torch_tangent_space_matches_numpy(metric):
    rng = np.random.RandomState(140)
    X = _make_spd((4,), 3, rng)
    Cref = _make_spd((), 3, rng)
    _assert_same_result(tangent_space, X, Cref, metric=metric)


@pytest.mark.parametrize("metric", TANGENT_METRICS)
def test_torch_untangent_space_matches_numpy(metric):
    rng = np.random.RandomState(141)
    X = _make_spd((4,), 3, rng)
    Cref = _make_spd((), 3, rng)
    T = tangent_space(X, Cref, metric=metric)
    tol = ITER_TOL if metric == "wasserstein" else STRICT_TOL
    _assert_same_result(untangent_space, T, Cref, metric=metric, tol=tol)


def test_torch_backend_autograd_smoke():
    rng = np.random.RandomState(199)
    A = (
        _to_torch(_make_spd((), 3, rng))
        .clone()
        .detach()
        .requires_grad_(True)
    )
    B = (
        _to_torch(_make_spd((), 3, rng))
        .clone()
        .detach()
        .requires_grad_(True)
    )
    X = (
        _to_torch(_make_spd((4,), 3, rng))
        .clone()
        .detach()
        .requires_grad_(True)
    )

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
