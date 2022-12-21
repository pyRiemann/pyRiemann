import numpy as np
import pytest

from pyriemann.utils.utils import check_weights


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
def test_check_weights_none(n_matrices):
    w = check_weights(None, n_matrices)
    assert np.sum(w) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
def test_check_weights_vals(rndstate, n_matrices):
    w = rndstate.rand(n_matrices) + 1
    w = check_weights(w, n_matrices)
    assert np.sum(w) == pytest.approx(1.0, abs=1e-10)


def test_check_weights_error_length():
    n_matrices = 5
    with pytest.raises(ValueError):  # not same length
        check_weights(np.ones(n_matrices), n_matrices + 1)


def test_check_weights_error_positivity():
    n_matrices = 5
    w = np.ones(n_matrices)
    w[0] = 0
    with pytest.raises(ValueError):  # not strictly positive weight
        check_weights(w, n_matrices, check_positivity=True)
