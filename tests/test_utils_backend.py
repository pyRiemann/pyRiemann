"""Tests for pyriemann.utils._backend module."""

import numpy as np
import pytest

from conftest import to_backend, to_numpy
from pyriemann.utils._backend import (
    as_numpy,
    from_numpy,
    diag_indices,
    tril_indices,
    triu_indices,
    get_namespace,
    xpd,
)


n_channels = 3


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
