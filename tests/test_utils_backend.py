"""Tests for pyriemann.utils._backend module."""

import numpy as np
import pytest
from conftest import to_backend, to_numpy

from pyriemann.utils._backend import (
    check_matrix_pair,
    to_numpy as backend_to_numpy,
    from_numpy,
    eigvalsh,
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
    like = to_backend(np.eye(3), backend)
    X = from_numpy(X_np, like=like)
    xp = get_namespace(X)
    assert xp.all(X == xp.asarray(X_np, dtype=like.dtype, device=xpd(like)))


def test_check_matrix_pair(backend):
    A = to_backend(np.random.rand(4, 3, 3), backend)
    B = to_backend(np.random.rand(4, 3, 3), backend)
    xp = check_matrix_pair(A, B)
    assert xp is get_namespace(A)


def test_check_matrix_pair_errors(backend):
    A = to_backend(np.random.rand(3, 3), backend)
    B = to_backend(np.random.rand(3, 4), backend)
    with pytest.raises(ValueError):
        check_matrix_pair(A, B)

    A1d = to_backend(np.random.rand(3), backend)
    with pytest.raises(ValueError):
        check_matrix_pair(A1d, A)


def test_check_matrix_pair_square(backend):
    A = to_backend(np.random.rand(3, 4), backend)
    B = to_backend(np.random.rand(3, 4), backend)
    with pytest.raises(ValueError):
        check_matrix_pair(A, B, require_square=True)


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


def test_eigvalsh(backend):
    rng = np.random.RandomState(42)
    A_np = rng.rand(3, 3)
    A_np = A_np @ A_np.T + np.eye(3)
    B_np = rng.rand(3, 3)
    B_np = B_np @ B_np.T + np.eye(3)
    A = to_backend(A_np, backend)
    B = to_backend(B_np, backend)
    ev = eigvalsh(A, B)
    assert to_numpy(ev).shape == (3,)

    from scipy.linalg import eigvalsh as sp_eigvalsh
    ev_ref = sp_eigvalsh(A_np, B_np)
    np.testing.assert_allclose(to_numpy(ev), ev_ref, atol=1e-10)


def test_eigvalsh_batched(backend):
    rng = np.random.RandomState(42)
    A_np = rng.rand(2, 3, 3)
    A_np = A_np @ np.swapaxes(A_np, -2, -1) + np.eye(3)
    B_np = rng.rand(2, 3, 3)
    B_np = B_np @ np.swapaxes(B_np, -2, -1) + np.eye(3)
    A = to_backend(A_np, backend)
    B = to_backend(B_np, backend)
    ev = eigvalsh(A, B)
    assert to_numpy(ev).shape == (2, 3)


def test_pairwise_euclidean(backend):
    rng = np.random.RandomState(42)
    X_np = rng.rand(5, 4)
    Y_np = rng.rand(3, 4)
    X = to_backend(X_np, backend)
    Y = to_backend(Y_np, backend)
    D = pairwise_euclidean(X, Y)
    assert to_numpy(D).shape == (5, 3)

    from scipy.spatial.distance import cdist
    D_ref = cdist(X_np, Y_np, metric='euclidean')
    np.testing.assert_allclose(to_numpy(D), D_ref, atol=1e-10)
