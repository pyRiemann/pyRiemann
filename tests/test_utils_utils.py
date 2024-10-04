import numpy as np
import pytest

from pyriemann.utils.utils import (
    check_weights,
    check_metric,
    check_function,
    check_init,
)


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
def test_check_weights_none(n_matrices):
    w = check_weights(None, n_matrices)
    assert np.sum(w) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
def test_check_weights_vals(get_weights, n_matrices):
    weights = get_weights(n_matrices) + 1
    weights = check_weights(weights, n_matrices)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-10)


def test_check_weights_error_length(get_weights):
    n_matrices = 5
    with pytest.raises(ValueError):  # not same length
        check_weights(get_weights(n_matrices), n_matrices + 1)


def test_check_weights_error_positivity(get_weights):
    n_matrices = 5
    weights = get_weights(n_matrices)
    weights[0] = 0
    with pytest.raises(ValueError):  # not strictly positive weight
        check_weights(weights, n_matrices, check_positivity=True)


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
    with pytest.raises(ValueError):  # not array
        check_init(init="init", n=3)
    with pytest.raises(ValueError):  # not 2D array
        check_init(init=np.ones((3, 2, 2)), n=3)
    with pytest.raises(ValueError):  # not 2D array
        check_init(init=[1, 2, 3], n=3)
    with pytest.raises(ValueError):  # not 2D array
        check_init(init=1, n=3)
    with pytest.raises(ValueError):  # not square array
        check_init(init=np.ones((3, 2)), n=3)
    with pytest.raises(ValueError):  # shape not equal to n
        check_init(init=np.ones((2, 2)), n=3)
