import numpy as np

from pyriemann.utils.test import (
    is_square, is_sym, is_skew_sym, is_real, is_hermitian,
    is_pos_def, is_pos_semi_def,
    is_sym_pos_def, is_sym_pos_semi_def,
    is_herm_pos_def, is_herm_pos_semi_def,
)

import pytest


n_channels = 3


def test_is_square():
    assert is_square(np.eye(n_channels))
    assert not is_square(np.ones((n_channels, n_channels + 1)))


def test_is_sym(rndstate):
    assert is_sym(np.eye(n_channels))
    A = rndstate.randn(n_channels, n_channels)
    assert is_sym(A + A.T)
    assert not is_sym(-A + A.T)
    assert not is_sym(np.ones((n_channels, n_channels + 1)))


def test_is_skew_sym(rndstate):
    A = rndstate.randn(n_channels, n_channels)
    assert is_skew_sym(A - A.T)
    assert not is_skew_sym(np.eye(n_channels))
    assert not is_skew_sym(A + A.T)
    assert not is_skew_sym(np.ones((n_channels, n_channels + 1)))


def test_is_real(rndstate):
    A = rndstate.randn(n_channels, n_channels + 2)
    assert is_real(A)

    B = np.zeros((n_channels, n_channels + 2), dtype=complex)
    B.real = A
    assert is_real(B)
    B.imag = A / 1000.0
    assert not is_real(B)


def test_is_hermitian(rndstate):
    A = rndstate.randn(n_channels, n_channels)
    B = np.zeros((n_channels, n_channels), dtype=complex)
    B.real, B.imag = A + A.T, A - A.T
    assert is_hermitian(B)
    assert not is_hermitian(np.ones((n_channels, n_channels + 1)))


@pytest.mark.parametrize("fast_mode", [True, False])
def test_is_pos_def(rndstate, fast_mode):
    assert is_pos_def(np.eye(n_channels), fast_mode=fast_mode)
    A = rndstate.randn(n_channels, 100)
    assert is_pos_def(A @ A.T + 0.01 * np.eye(n_channels), fast_mode=fast_mode)
    assert not is_pos_def(-A @ A.T, fast_mode=fast_mode)
    assert not is_pos_def(np.ones((n_channels, n_channels + 1)),
                          fast_mode=fast_mode)


def test_is_pos_semi_def(rndstate):
    assert is_pos_semi_def(np.eye(n_channels))
    A = rndstate.randn(n_channels, 100)
    assert is_pos_semi_def(A @ A.T)
    assert not is_pos_semi_def(-A @ A.T)
    assert not is_pos_semi_def(np.ones((n_channels, n_channels + 1)))


def test_is_sym_pos_def(rndstate):
    assert is_sym_pos_def(np.eye(n_channels))
    A = rndstate.randn(n_channels, 100)
    assert is_sym_pos_def(A @ A.T + 0.001 * np.eye(n_channels))
    assert not is_sym_pos_def(-A @ A.T)
    assert not is_sym_pos_def(np.ones((n_channels, n_channels + 1)))

    B = A - np.mean(A, axis=0)
    assert not is_sym_pos_def(B @ B.T, tol=1e-9)


def test_is_sym_pos_semi_def(rndstate):
    assert is_sym_pos_semi_def(np.eye(n_channels))
    A = rndstate.randn(n_channels, 100)
    assert is_sym_pos_semi_def(A @ A.T)
    assert not is_sym_pos_semi_def(-A @ A.T)
    assert not is_sym_pos_semi_def(np.ones((n_channels, n_channels + 1)))

    B = A - np.mean(A, axis=0)
    assert is_sym_pos_semi_def(B @ B.T)


def test_is_herm_pos_def(rndstate):
    assert is_herm_pos_def(np.eye(n_channels))
    A = rndstate.randn(n_channels, 100) + 1j * rndstate.randn(n_channels, 100)
    assert is_herm_pos_def(A @ A.conj().T + 0.001 * np.eye(n_channels))
    assert not is_herm_pos_def(A @ A.T)  # pseudo-covariance is not HPD
    assert not is_herm_pos_def(-A @ A.conj().T)
    assert not is_herm_pos_def(np.ones((n_channels, n_channels + 1)))

    B = A - np.mean(A, axis=0)
    assert not is_herm_pos_def(B @ B.conj().T, tol=1e-9)


def test_is_herm_pos_semi_def(rndstate):
    assert is_herm_pos_semi_def(np.eye(n_channels))
    A = rndstate.randn(n_channels, 100) + 1j * rndstate.randn(n_channels, 100)
    assert is_herm_pos_semi_def(A @ A.conj().T)
    assert not is_herm_pos_semi_def(A @ A.T)  # pseudo-covariance is not HPSD
    assert not is_herm_pos_semi_def(-A @ A.conj().T)
    assert not is_herm_pos_semi_def(np.ones((n_channels, n_channels + 1)))

    B = A - np.mean(A, axis=0)
    assert is_herm_pos_semi_def(B @ B.conj().T)
