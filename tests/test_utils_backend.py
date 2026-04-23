"""Tests for pyriemann.utils._backend module."""

import numpy as np

from conftest import to_backend, to_numpy
from pyriemann.utils._backend import (
    as_numpy,
    from_numpy,
    diag_indices,
    hann_window,
    nanmean,
    tril_indices,
    triu_indices,
    get_namespace,
    xpd,
)


def test_as_numpy():
    X = np.random.rand(3, 3)
    assert np.array_equal(as_numpy(X), X)


def test_from_numpy(backend):
    X_np = np.random.rand(3, 3)
    like = to_backend(np.eye(3), backend)
    X = from_numpy(X_np, like=like)
    xp = get_namespace(X)
    assert xp.all(X == xp.asarray(X_np, dtype=like.dtype, device=xpd(like)))


def test_diag_indices(backend):
    like = to_backend(np.eye(3), backend)
    idx0, idx1 = diag_indices(3, like=like)
    assert len(to_numpy(idx0)) == 3
    assert len(to_numpy(idx1)) == 3


def test_tril_indices(backend):
    like = to_backend(np.eye(4), backend)
    idx0, idx1 = tril_indices(4, -1, like=like)
    n_expected = 4 * 3 // 2  # n*(n-1)/2 for strict lower
    assert len(to_numpy(idx0)) == n_expected


def test_triu_indices(backend):
    like = to_backend(np.eye(4), backend)
    idx0, idx1 = triu_indices(4, 1, like=like)
    n_expected = 4 * 3 // 2  # n*(n-1)/2 for strict upper
    assert len(to_numpy(idx0)) == n_expected


def test_hann_window(backend):
    like = to_backend(np.zeros(1, dtype=np.float64), backend)
    win = hann_window(8, like=like)
    np.testing.assert_array_almost_equal(to_numpy(win), np.hanning(8))
    # n=1 edge case returns ones
    win1 = hann_window(1, like=like)
    assert win1.shape == (1,)
    np.testing.assert_array_equal(to_numpy(win1), np.array([1.0]))


def test_nanmean(backend):
    # With no NaN, equivalent to xp.mean
    X_np = np.arange(12, dtype=np.float64).reshape(3, 4)
    X = to_backend(X_np, backend)
    xp = get_namespace(X)
    np.testing.assert_array_almost_equal(
        to_numpy(nanmean(X, axis=0)), to_numpy(xp.mean(X, axis=0)),
    )
    np.testing.assert_array_almost_equal(
        to_numpy(nanmean(X, axis=1)), to_numpy(xp.mean(X, axis=1)),
    )
    # With NaN, equivalent to np.nanmean
    X_nan_np = X_np.copy()
    X_nan_np[0, 1] = np.nan
    X_nan_np[2, 3] = np.nan
    X_nan = to_backend(X_nan_np, backend)
    np.testing.assert_array_almost_equal(
        to_numpy(nanmean(X_nan, axis=0)), np.nanmean(X_nan_np, axis=0),
    )
