"""Broadcast compatibility tests for pyriemann utility functions.

Tests all utility functions with batched inputs (4D and 5D).
Functions not yet broadcast-compatible are marked xfail(strict=True),
serving as a regression checklist: remove the marker once fixed.
"""

from functools import partial

import numpy as np
import pytest

from pyriemann.datasets import make_matrices
from pyriemann.utils.base import (
    ddexpm, ddlogm, expm, invsqrtm, logm,
    nearest_sym_pos_def, powm, sqrtm,
)
from pyriemann.utils.distance import (
    distance_chol, distance_euclid, distance_harmonic,
    distance_kullback, distance_kullback_right, distance_kullback_sym,
    distance_logchol, distance_logdet, distance_logeuclid,
    distance_poweuclid, distance_riemann, distance_thompson,
    distance_wasserstein,
)
from pyriemann.utils.geodesic import (
    geodesic_chol, geodesic_euclid, geodesic_logchol,
    geodesic_logeuclid, geodesic_riemann, geodesic_thompson,
    geodesic_wasserstein,
)
from pyriemann.utils.mean import (
    maskedmean_riemann, mean_ale, mean_alm, mean_chol, mean_euclid,
    mean_harmonic, mean_logchol, mean_logdet, mean_logeuclid,
    mean_power, mean_poweuclid, mean_riemann, mean_thompson,
    mean_wasserstein, nanmean_riemann,
)
from pyriemann.utils.tangentspace import (
    exp_map_logchol, exp_map_riemann, log_map_logchol, log_map_riemann,
    tangent_space, transport_logchol, transport_logeuclid,
    transport_riemann, unupper, untangent_space, upper,
)
from pyriemann.utils.covariance import (
    block_covariances, coherence, cospectrum, covariance_mest,
    covariance_sch, covariance_scm, covariances, covariances_EP,
    covariances_X, cross_spectrum, get_nondiag_weight, normalize,
)

# ---- constants ----

N_DIM = 3
N_VEC = N_DIM * (N_DIM + 1) // 2  # 6
N_MAT = 5  # number of matrices for mean tests
BATCH_SHAPES = [(2, 3), (2, 3, 4)]

_xfail = pytest.mark.xfail(strict=True, reason="not yet broadcast-compatible")


# ---- helpers ----

def _make_batch_spd(batch_shape, n_dim=N_DIM, seed=42):
    """Generate SPD matrices with shape (*batch_shape, n_dim, n_dim)."""
    rs = np.random.RandomState(seed)
    n_total = int(np.prod(batch_shape))
    flat = make_matrices(n_total, n_dim, "spd", rs, return_params=False)
    return flat.reshape(*batch_shape, n_dim, n_dim)


def _make_single_spd(n_dim=N_DIM, seed=99):
    """Generate a single SPD matrix of shape (n_dim, n_dim)."""
    rs = np.random.RandomState(seed)
    return make_matrices(1, n_dim, "spd", rs, return_params=False)[0]


def _first(batch_shape):
    """Index tuple selecting the first element across all batch dims."""
    return (0,) * len(batch_shape)


def _mean_first(batch_shape):
    """Index selecting X[:, 0, 0, ...] — first batch element, all matrices."""
    return (slice(None),) + (0,) * len(batch_shape)


# ===========================================================
# Base unary: f(X) -> (*batch, n, n)
# ===========================================================

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func", [
    sqrtm,
    invsqrtm,
    logm,
    expm,
    pytest.param(partial(powm, alpha=0.5), id="powm"),
    pytest.param(nearest_sym_pos_def, id="nearest_sym_pos_def"),
])
def test_base_unary(func, batch_shape):
    X = _make_batch_spd(batch_shape)
    result = func(X)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], func(X[idx]), atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func", [ddexpm, ddlogm, log_map_riemann, log_map_logchol])
def test_unary_with_ref(func, batch_shape):
    """Functions taking (X, Cref) where Cref is a 2D reference matrix."""
    X = _make_batch_spd(batch_shape)
    Cref = _make_single_spd()
    result = func(X, Cref)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], func(X[idx], Cref), atol=1e-10)


# ===========================================================
# Distance pair: f(A, B) -> (*batch,)
# ===========================================================

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func", [
    distance_chol,
    distance_euclid,
    distance_harmonic,
    distance_logchol,
    distance_logeuclid,
    distance_riemann,
    distance_wasserstein,
    distance_kullback,
    distance_kullback_right,
    distance_kullback_sym,
    distance_logdet,
    distance_thompson,
    pytest.param(partial(distance_poweuclid, p=0.5), id="distance_poweuclid"),
])
def test_distance_pair(func, batch_shape):
    A = _make_batch_spd(batch_shape, seed=42)
    B = _make_batch_spd(batch_shape, seed=7)
    result = func(A, B)
    assert result.shape == batch_shape
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], func(A[idx], B[idx]), atol=1e-10)


# ===========================================================
# Geodesic pair: f(A, B) -> (*batch, n, n)
# ===========================================================

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func", [
    geodesic_chol,
    geodesic_euclid,
    geodesic_logchol,
    geodesic_logeuclid,
    geodesic_riemann,
    geodesic_wasserstein,
    geodesic_thompson,
])
def test_geodesic_pair(func, batch_shape):
    A = _make_batch_spd(batch_shape, seed=42)
    B = _make_batch_spd(batch_shape, seed=7)
    result = func(A, B)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], func(A[idx], B[idx]), atol=1e-10)


# ===========================================================
# Exp maps: f(T, Cref) -> (*batch, n, n)
# ===========================================================

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func, log_func", [
    pytest.param(exp_map_riemann, log_map_riemann, id="exp_map_riemann"),
    pytest.param(exp_map_logchol, log_map_logchol, id="exp_map_logchol"),
])
def test_exp_map(func, log_func, batch_shape):
    X = _make_batch_spd(batch_shape, seed=42)
    Cref = _make_single_spd()
    T = log_func(X, Cref)  # generate valid tangent vectors
    result = func(T, Cref)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], func(T[idx], Cref), atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func", [
    transport_riemann,
    transport_logeuclid,
    transport_logchol,
])
def test_transport(func, batch_shape):
    X = _make_batch_spd(batch_shape, seed=42)
    A = _make_batch_spd(batch_shape, seed=7)
    B = _make_batch_spd(batch_shape, seed=13)
    result = func(X, A, B)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    np.testing.assert_allclose(
        result[idx], func(X[idx], A[idx], B[idx]), atol=1e-10
    )


# ===========================================================
# Upper / tangent space
# ===========================================================

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_upper(batch_shape):
    X = _make_batch_spd(batch_shape)
    result = upper(X)
    assert result.shape == (*batch_shape, N_VEC)
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], upper(X[idx]), atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_unupper(batch_shape):
    X = _make_batch_spd(batch_shape)
    T = upper(X)
    result = unupper(T)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], unupper(T[idx]), atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_tangent_space(batch_shape):
    X = _make_batch_spd(batch_shape, seed=42)
    Cref = _make_single_spd()
    result = tangent_space(X, Cref)
    assert result.shape == (*batch_shape, N_VEC)
    idx = _first(batch_shape)
    np.testing.assert_allclose(
        result[idx], tangent_space(X[idx], Cref), atol=1e-10
    )


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_untangent_space(batch_shape):
    X = _make_batch_spd(batch_shape, seed=42)
    Cref = _make_single_spd()
    T = tangent_space(X, Cref)
    result = untangent_space(T, Cref)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    np.testing.assert_allclose(
        result[idx], untangent_space(T[idx], Cref), atol=1e-10
    )


# ===========================================================
# Means: f(X) -> (*batch, n, n), reducing axis 0
# ===========================================================

def _make_mean_input(batch_shape, seed=42):
    """Shape: (N_MAT, *batch_shape, N_DIM, N_DIM)."""
    return _make_batch_spd((N_MAT, *batch_shape), seed=seed)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func", [
    mean_euclid,
    mean_harmonic,
    mean_logeuclid,
    pytest.param(partial(mean_poweuclid, p=0.5), id="mean_poweuclid"),
    mean_chol,
    mean_riemann,
    mean_wasserstein,
    mean_logchol,
    pytest.param(mean_ale, id="mean_ale"),
    pytest.param(mean_alm, id="mean_alm"),
    # tol=1e-12 tightens convergence so batched and single-element results match
    pytest.param(partial(mean_logdet, tol=1e-12), id="mean_logdet"),
    mean_thompson,
    pytest.param(partial(mean_power, p=0.5), id="mean_power"),
    pytest.param(nanmean_riemann, id="nanmean_riemann"),
])
def test_mean(func, batch_shape):
    X = _make_mean_input(batch_shape)
    result = func(X)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    idx_slice = _mean_first(batch_shape)
    np.testing.assert_allclose(result[idx], func(X[idx_slice]), atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_maskedmean_riemann(batch_shape):
    X = _make_mean_input(batch_shape)
    masks = [np.eye(N_DIM)] * N_MAT  # identity masks = all channels present
    result = maskedmean_riemann(X, masks)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    idx_slice = _mean_first(batch_shape)
    np.testing.assert_allclose(
        result[idx], maskedmean_riemann(X[idx_slice], masks), atol=1e-10
    )


# ===========================================================
# Covariance utilities
# ===========================================================

N_CHANNELS = 3
N_TIMES = 128
SPEC_WINDOW = 64
N_FREQS = SPEC_WINDOW // 2 + 1


def _make_batch_ts(batch_shape, n_channels=N_CHANNELS, n_times=N_TIMES,
                   seed=42):
    """Generate time-series: (*batch_shape, n_channels, n_times)."""
    rs = np.random.RandomState(seed)
    return rs.randn(*batch_shape, n_channels, n_times)


# -----------------------------------------------------------
# Single-trial estimators: (*batch, n_ch, n_t) -> (*batch, n_ch, n_ch)
# -----------------------------------------------------------

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func", [
    covariance_scm,
    covariance_sch,
])
def test_cov_single_trial(func, batch_shape):
    X = _make_batch_ts(batch_shape)
    result = func(X)
    assert result.shape == (*batch_shape, N_CHANNELS, N_CHANNELS)
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], func(X[idx]), atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("m_est", ["hub", "stu", "tyl"])
def test_cov_mest(m_est, batch_shape):
    X = _make_batch_ts(batch_shape)
    result = covariance_mest(X, m_est)
    assert result.shape == (*batch_shape, N_CHANNELS, N_CHANNELS)
    idx = _first(batch_shape)
    np.testing.assert_allclose(
        result[idx], covariance_mest(X[idx], m_est), atol=1e-6
    )


# -----------------------------------------------------------
# Multi-trial wrappers: (*batch, n_ch, n_t) -> (*batch, n_ch_out, n_ch_out)
# -----------------------------------------------------------

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_covariances(batch_shape):
    X = _make_batch_ts(batch_shape)
    result = covariances(X)
    assert result.shape == (*batch_shape, N_CHANNELS, N_CHANNELS)
    idx = _first(batch_shape)
    np.testing.assert_allclose(result[idx], covariances(X[idx]), atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_covariances_EP(batch_shape):
    X = _make_batch_ts(batch_shape)
    P = np.random.RandomState(77).randn(2, N_TIMES)
    n_out = N_CHANNELS + 2
    result = covariances_EP(X, P)
    assert result.shape == (*batch_shape, n_out, n_out)
    idx = _first(batch_shape)
    ref = covariances_EP(X[idx][np.newaxis], P)[0]
    np.testing.assert_allclose(result[idx], ref, atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_covariances_X(batch_shape):
    X = _make_batch_ts(batch_shape)
    n_out = N_CHANNELS + N_TIMES
    result = covariances_X(X)
    assert result.shape == (*batch_shape, n_out, n_out)
    idx = _first(batch_shape)
    ref = covariances_X(X[idx][np.newaxis])[0]
    np.testing.assert_allclose(result[idx], ref, atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_block_covariances(batch_shape):
    X = _make_batch_ts(batch_shape)
    blocks = [1, 2]  # must sum to N_CHANNELS
    result = block_covariances(X, blocks)
    assert result.shape == (*batch_shape, N_CHANNELS, N_CHANNELS)
    idx = _first(batch_shape)
    ref = block_covariances(X[idx][np.newaxis], blocks)[0]
    np.testing.assert_allclose(result[idx], ref, atol=1e-10)


# -----------------------------------------------------------
# Spectral: (*batch, n_ch, n_t) -> (*batch, n_ch, n_ch, n_freqs)
# -----------------------------------------------------------

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("func", [
    cross_spectrum,
    cospectrum,
])
def test_spectral(func, batch_shape):
    X = _make_batch_ts(batch_shape)
    S, freqs = func(X, window=SPEC_WINDOW)
    assert S.shape == (*batch_shape, N_CHANNELS, N_CHANNELS, N_FREQS)
    idx = _first(batch_shape)
    S_ref, _ = func(X[idx], window=SPEC_WINDOW)
    np.testing.assert_allclose(S[idx], S_ref, atol=1e-10)


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("coh_type", [
    "ordinary", "instantaneous", "lagged", "imaginary",
])
def test_coherence(coh_type, batch_shape):
    X = _make_batch_ts(batch_shape)
    C, freqs = coherence(X, window=SPEC_WINDOW, coh=coh_type)
    assert C.shape == (*batch_shape, N_CHANNELS, N_CHANNELS, N_FREQS)
    idx = _first(batch_shape)
    C_ref, _ = coherence(X[idx], window=SPEC_WINDOW, coh=coh_type)
    np.testing.assert_allclose(C[idx], C_ref, atol=1e-10)


# -----------------------------------------------------------
# Normalize / get_nondiag_weight: (*batch, n, n) -> (*batch, n, n) or (*batch,)
# -----------------------------------------------------------

@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
@pytest.mark.parametrize("norm_type", ["corr", "trace", "determinant"])
def test_normalize(norm_type, batch_shape):
    X = _make_batch_spd(batch_shape)
    result = normalize(X, norm_type)
    assert result.shape == (*batch_shape, N_DIM, N_DIM)
    idx = _first(batch_shape)
    np.testing.assert_allclose(
        result[idx], normalize(X[idx], norm_type), atol=1e-10
    )


@pytest.mark.parametrize("batch_shape", BATCH_SHAPES)
def test_get_nondiag_weight(batch_shape):
    X = _make_batch_spd(batch_shape)
    result = get_nondiag_weight(X)
    assert result.shape == batch_shape
    idx = _first(batch_shape)
    np.testing.assert_allclose(
        result[idx], get_nondiag_weight(X[idx]), atol=1e-10
    )
