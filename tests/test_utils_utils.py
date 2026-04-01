import warnings

import numpy as np
import pytest

from conftest import _to_backend
from pyriemann.datasets import make_matrices
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.geodesic import geodesic_logchol
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import log_map_riemann, exp_map_riemann
from pyriemann.utils.utils import (
    check_weights,
    check_metric,
    check_function,
    check_init,
)


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
def test_check_weights_none(n_matrices):
    like = np.ones(n_matrices)
    w = check_weights(None, n_matrices, like=like)
    assert np.sum(w) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("n_matrices", [3, 4, 5])
def test_check_weights_vals(get_weights, n_matrices):
    weights = get_weights(n_matrices) + 1
    weights = check_weights(weights, n_matrices, like=weights)
    assert np.sum(weights) == pytest.approx(1.0, abs=1e-10)


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
        check_weights(weights, n_matrices,
                      check_positivity=True, like=weights)


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
    with pytest.raises(ValueError):  # not array
        check_init(init="init", n=3, like=like)
    with pytest.raises(ValueError):  # not 2D array
        check_init(init=np.ones((3, 2, 2)), n=3, like=like)
    with pytest.raises(ValueError):  # not 2D array
        check_init(init=[1, 2, 3], n=3, like=like)
    with pytest.raises(ValueError):  # not 2D array
        check_init(init=1, n=3, like=like)
    with pytest.raises(ValueError):  # not square array
        check_init(init=np.ones((3, 2)), n=3, like=like)
    with pytest.raises(ValueError):  # shape not equal to n
        check_init(init=np.ones((2, 2)), n=3, like=like)


def test_autograd_smoke():
    torch = pytest.importorskip("torch")
    torch.set_grad_enabled(True)  # re-enable for this test
    rng = np.random.RandomState(199)

    A_np, B_np = make_matrices(2, 3, "spd", rng, return_params=False)
    X_np = make_matrices(4, 3, "spd", rng, return_params=False)

    A = _to_backend(A_np, "torch").clone().detach().requires_grad_(True)
    B = _to_backend(B_np, "torch").clone().detach().requires_grad_(True)
    X = _to_backend(X_np, "torch").clone().detach().requires_grad_(True)

    tangent = log_map_riemann(B.unsqueeze(0), A, C12=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        loss = distance_riemann(A, B)
        loss = loss + geodesic_logchol(A, B, alpha=0.25).sum()
        loss = loss + mean_riemann(X, maxiter=5).sum()
        loss = loss + exp_map_riemann(tangent, A, Cm12=True).sum()
    loss.backward()

    for grad in (A.grad, B.grad, X.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
