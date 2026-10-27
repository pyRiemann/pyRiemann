from array_api_compat import array_namespace as get_namespace
import numpy as np
import pytest
from pytest import approx

from conftest import to_backend
from pyriemann.utils._check import (
    check_function,
    check_init,
    check_matrix_pair,
    check_metric,
    check_weights,
)


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
def test_check_weights_none(n_matrices):
    weights = check_weights(None, n_matrices)
    xp = get_namespace(weights)
    assert weights.dtype == float
    assert weights.shape == (n_matrices,)
    assert xp.sum(weights) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
def test_check_weights_vals(get_weights, n_matrices):
    weights = get_weights(n_matrices) + 1
    weights = check_weights(weights, n_matrices, like=weights)
    xp = get_namespace(weights)
    assert weights.shape == (n_matrices,)
    assert xp.sum(weights) == pytest.approx(1.0, abs=1e-10)


def test_check_weights_error_length(get_weights):
    n_matrices = 5
    weights = get_weights(n_matrices)
    with pytest.raises(ValueError):  # not same length
        check_weights(weights, n_matrices + 1, like=weights)


def test_check_weights_error_positivity(get_weights):
    n_matrices = 5
    weights = get_weights(n_matrices)
    weights[0] = 0
    with pytest.raises(ValueError):  # not strictly positive weight
        check_weights(weights, n_matrices, check_positivity=True, like=weights)


def test_check_metric_str():
    given_metric = "abc"
    expected_keys = ["mean", "distance"]
    metrics = check_metric(given_metric, expected_keys)
    assert len(metrics) == len(expected_keys)


def test_check_metric_dict():
    given_dict = {"mean": "aaa", "distance": "bbb"}
    expected_keys = ["mean", "distance"]
    metrics = check_metric(given_dict, expected_keys)
    assert len(metrics) == len(expected_keys)

    with pytest.raises(KeyError):
        check_metric({"mean": "aaa", "map": "bbb"}, expected_keys)


def test_check_metric_errortype():
    with pytest.raises(TypeError):
        check_metric(3)


def test_check_function():
    def aaa(): return True
    def bbb(): return False
    available_funs = {"aaa": aaa, "bbb": bbb}

    fun = check_function("aaa", available_funs)
    assert hasattr(fun, "__call__")

    fun = check_function(aaa, available_funs)
    assert hasattr(fun, "__call__")

    with pytest.raises(ValueError):  # unkown function name
        check_function("abc", available_funs)

    with pytest.raises(ValueError):  # not str or callable
        check_function(0.5, available_funs)


def test_check_init():
    like = np.ones((3, 3))
    with pytest.raises(AttributeError):  # not array
        check_init(init="init", n=3, like=like)
    with pytest.raises(ValueError):  # not 2D array
        check_init(init=np.ones((3, 2, 2)), n=3, like=like)
    with pytest.raises(AttributeError):  # not 2D array
        check_init(init=[1, 2, 3], n=3, like=like)
    with pytest.raises(AttributeError):  # not 2D array
        check_init(init=1, n=3, like=like)
    with pytest.raises(ValueError):  # not square array
        check_init(init=np.ones((3, 2)), n=3, like=like)
    with pytest.raises(ValueError):  # shape not equal to n
        check_init(init=np.ones((2, 2)), n=3, like=like)


@pytest.mark.parametrize("kind", ["real", "comp"])
def test_check_init_copy(kind, get_mats):
    """Test that output is a copy of input"""
    n = 3
    init = get_mats(1, n, kind=kind)[0]
    init_checked = check_init(init, n, like=init)

    assert init_checked.dtype == init.dtype
    assert init_checked == approx(init)
    init_checked[0, 0] = 42  # modifying output does not modify input
    assert init[0, 0] != 42


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
