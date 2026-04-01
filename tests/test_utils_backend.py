"""Tests for pyriemann.utils._backend module."""

import numpy as np
import pytest
from conftest import _to_backend, to_numpy

from pyriemann.utils._backend import (
    check_matrix_pair,
    to_numpy as backend_to_numpy,
    from_numpy,
    weighted_average,
    joint_eigvalsh,
    pairwise_euclidean,
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


def test_joint_eigvalsh(backend):
    rng = np.random.RandomState(42)
    A_np = rng.rand(3, 3)
    A_np = A_np @ A_np.T + np.eye(3)
    B_np = rng.rand(3, 3)
    B_np = B_np @ B_np.T + np.eye(3)
    A = _to_backend(A_np, backend)
    B = _to_backend(B_np, backend)
    xp = get_namespace(A)
    ev = joint_eigvalsh(A, B, xp=xp)
    assert to_numpy(ev).shape == (3,)

    from scipy.linalg import eigvalsh as sp_eigvalsh
    ev_ref = sp_eigvalsh(A_np, B_np)
    np.testing.assert_allclose(to_numpy(ev), ev_ref, atol=1e-10)


def test_joint_eigvalsh_batched(backend):
    rng = np.random.RandomState(42)
    A_np = rng.rand(2, 3, 3)
    A_np = A_np @ np.swapaxes(A_np, -2, -1) + np.eye(3)
    B_np = rng.rand(2, 3, 3)
    B_np = B_np @ np.swapaxes(B_np, -2, -1) + np.eye(3)
    A = _to_backend(A_np, backend)
    B = _to_backend(B_np, backend)
    xp = get_namespace(A)
    ev = joint_eigvalsh(A, B, xp=xp)
    assert to_numpy(ev).shape == (2, 3)


def test_pairwise_euclidean(backend):
    rng = np.random.RandomState(42)
    X_np = rng.rand(5, 4)
    Y_np = rng.rand(3, 4)
    X = _to_backend(X_np, backend)
    Y = _to_backend(Y_np, backend)
    xp = get_namespace(X)
    D = pairwise_euclidean(X, Y, xp=xp)
    assert to_numpy(D).shape == (5, 3)

    from scipy.spatial.distance import cdist
    D_ref = cdist(X_np, Y_np, metric='euclidean')
    np.testing.assert_allclose(to_numpy(D), D_ref, atol=1e-10)
