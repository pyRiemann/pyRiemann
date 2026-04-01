"""Tests for pyriemann.utils._backend module."""

import numpy as np
import pytest
from conftest import _to_backend, to_numpy

from pyriemann.utils._backend import (
    check_matrix_pair,
    to_numpy as backend_to_numpy,
    from_numpy,
    weighted_average,
    diag_indices,
    tril_indices,
    triu_indices,
    get_namespace,
    xpd,
)


n_channels = 3


def test_to_numpy():
    X = np.random.rand(3, 3)
    assert np.array_equal(backend_to_numpy(X), X)


def test_from_numpy(backend):
    X_np = np.random.rand(3, 3)
    like = _to_backend(np.eye(3), backend)
    X = from_numpy(X_np, like=like)
    xp = get_namespace(X)
    assert xp.all(X == xp.asarray(X_np, dtype=like.dtype, device=xpd(like)))


def test_check_matrix_pair(backend):
    A = _to_backend(np.random.rand(4, 3, 3), backend)
    B = _to_backend(np.random.rand(4, 3, 3), backend)
    xp = check_matrix_pair(A, B)
    assert xp is get_namespace(A)


def test_check_matrix_pair_errors(backend):
    A = _to_backend(np.random.rand(3, 3), backend)
    B = _to_backend(np.random.rand(3, 4), backend)
    with pytest.raises(ValueError):
        check_matrix_pair(A, B)

    A1d = _to_backend(np.random.rand(3), backend)
    with pytest.raises(ValueError):
        check_matrix_pair(A1d, A)


def test_check_matrix_pair_square(backend):
    A = _to_backend(np.random.rand(3, 4), backend)
    B = _to_backend(np.random.rand(3, 4), backend)
    with pytest.raises(ValueError):
        check_matrix_pair(A, B, require_square=True)


def test_weighted_average(backend):
    X = _to_backend(np.random.rand(5, 3, 3), backend)
    xp = get_namespace(X)
    w = _to_backend(np.array([0.1, 0.2, 0.3, 0.2, 0.2]), backend)

    avg = weighted_average(X, weights=w, axis=0, xp=xp)
    assert avg.shape == (3, 3)

    avg_none = weighted_average(X, weights=None, axis=0, xp=xp)
    np.testing.assert_allclose(
        to_numpy(avg_none), to_numpy(xp.mean(X, axis=0)), atol=1e-10,
    )


def test_diag_indices(backend):
    like = _to_backend(np.eye(3), backend)
    xp = get_namespace(like)
    idx0, idx1 = diag_indices(3, xp=xp, like=like)
    assert len(to_numpy(idx0)) == 3
    assert len(to_numpy(idx1)) == 3


def test_tril_indices(backend):
    like = _to_backend(np.eye(4), backend)
    xp = get_namespace(like)
    idx0, idx1 = tril_indices(4, -1, xp=xp, like=like)
    n_expected = 4 * 3 // 2  # n*(n-1)/2 for strict lower
    assert len(to_numpy(idx0)) == n_expected


def test_triu_indices(backend):
    like = _to_backend(np.eye(4), backend)
    xp = get_namespace(like)
    idx0, idx1 = triu_indices(4, 1, xp=xp, like=like)
    n_expected = 4 * 3 // 2  # n*(n-1)/2 for strict upper
    assert len(to_numpy(idx0)) == n_expected
