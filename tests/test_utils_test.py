import numpy as np
import pytest

from pyriemann.utils.test import (
    is_square, is_sym, is_skew_sym, is_hankel,
    is_real, is_real_type, is_hermitian,
    is_pos_def, is_pos_semi_def,
    is_sym_pos_def, is_sym_pos_semi_def,
    is_herm_pos_def, is_herm_pos_semi_def,
)


n = 3


def test_is_square():
    assert is_square(np.eye(n))
    assert not is_square(np.ones((n, n + 1)))


def test_is_sym(rndstate):
    assert is_sym(np.eye(n))

    A = rndstate.randn(n, n)
    assert is_sym(A + A.T)
    assert not is_sym(-A + A.T)
    assert not is_sym(np.ones((n, n + 1)))


def test_is_skew_sym(rndstate):
    A = rndstate.randn(n, n)
    assert is_skew_sym(A - A.T)
    assert not is_skew_sym(np.eye(n))
    assert not is_skew_sym(A + A.T)
    assert not is_skew_sym(np.ones((n, n + 1)))


def test_is_hankel():
    assert is_hankel(np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]]))
    assert not is_hankel(np.array([[1, 2, 3], [1, 3, 4], [3, 4, 5]]))
    assert not is_hankel(np.array([[1, 2, 3], [2, 3, 3], [3, 4, 5]]))


def test_is_real(rndstate):
    A = rndstate.randn(n, n + 2)
    assert is_real(A)

    B = np.zeros((n, n + 2), dtype=complex)
    B.real = A
    assert is_real(B)
    B.imag = A / 1000.0
    assert not is_real(B)


def test_is_real_type(rndstate):
    A = np.zeros((n, n + 1))
    assert is_real_type(A)

    B = np.zeros((n, n + 2), dtype=complex)
    assert not is_real_type(B)


def test_is_hermitian(rndstate):
    A = rndstate.randn(n, n)
    B = np.zeros((n, n), dtype=complex)
    B.real, B.imag = A + A.T, A - A.T
    assert is_hermitian(B)
    assert not is_hermitian(np.ones((n, n)) + 1j * np.ones((n, n)))


@pytest.mark.parametrize("fast_mode", [True, False])
def test_is_pos_def(rndstate, fast_mode):
    assert is_pos_def(np.eye(n), fast_mode=fast_mode)

    A = rndstate.randn(n, 100)
    assert is_pos_def(A @ A.T + 1e-6 * np.eye(n), fast_mode=fast_mode)
    assert not is_pos_def(-A @ A.T, fast_mode=fast_mode)
    assert not is_pos_def(np.ones((n, n + 1)), fast_mode=fast_mode)


def test_is_pos_semi_def(rndstate):
    assert is_pos_semi_def(np.eye(n))

    A = rndstate.randn(n, 100)
    assert is_pos_semi_def(A @ A.T)
    assert not is_pos_semi_def(-A @ A.T)
    assert not is_pos_semi_def(np.ones((n, n + 1)))


def test_is_sym_pos_def(rndstate):
    assert is_sym_pos_def(np.eye(n))

    A = rndstate.randn(n, 100)
    assert is_sym_pos_def(A @ A.T + 1e-6 * np.eye(n))
    assert not is_sym_pos_def(-A @ A.T)
    assert not is_sym_pos_def(np.ones((n, n + 1)))

    B = A - np.mean(A, axis=0)
    assert not is_sym_pos_def(B @ B.T, tol=1e-9)


def test_is_sym_pos_semi_def(rndstate):
    assert is_sym_pos_semi_def(np.eye(n))

    A = rndstate.randn(n, 100)
    assert is_sym_pos_semi_def(A @ A.T)
    assert not is_sym_pos_semi_def(-A @ A.T)
    assert not is_sym_pos_semi_def(np.ones((n, n + 1)))

    B = A - np.mean(A, axis=0)
    assert is_sym_pos_semi_def(B @ B.T)


def test_is_herm_pos_def(rndstate):
    assert is_herm_pos_def(np.eye(n))

    A = rndstate.randn(n, 100) + 1j * rndstate.randn(n, 100)
    assert is_herm_pos_def(A @ A.conj().T + 1e-6 * np.eye(n))
    assert not is_herm_pos_def(A @ A.T)  # pseudo-covariance is not HPD
    assert not is_herm_pos_def(-A @ A.conj().T)
    assert not is_herm_pos_def(np.ones((n, n + 1)))

    B = A - np.mean(A, axis=0)
    assert not is_herm_pos_def(B @ B.conj().T, tol=1e-9)


def test_is_herm_pos_semi_def(rndstate):
    assert is_herm_pos_semi_def(np.eye(n))

    A = rndstate.randn(n, 100) + 1j * rndstate.randn(n, 100)
    assert is_herm_pos_semi_def(A @ A.conj().T)
    assert not is_herm_pos_semi_def(A @ A.T)  # pseudo-covariance is not HPSD
    assert not is_herm_pos_semi_def(-A @ A.conj().T)
    assert not is_herm_pos_semi_def(np.ones((n, n + 1)))

    B = A - np.mean(A, axis=0)
    assert is_herm_pos_semi_def(B @ B.conj().T + 1e-9 * np.eye(n))
