import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from pytest import approx

from pyriemann.geometry.distance import distance
from pyriemann.optimization.grassmann import (
    _get_rotation_manifold,
    _get_rotation_tangentspace,
    _grad,
    _loss,
)


pytestmark = pytest.mark.numpy_only


def _is_orth(X):
    return X @ X.T == approx(np.eye(X.shape[0]))


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
def test_grassmann_loss(metric, get_mats, get_weights):
    """Test that loss is the weighted sum of squared distances"""
    n_matrices, n_channels = 3, 4
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices)
    Q = get_mats(1, n_channels, "orth")[0]

    assert _loss(Q, X, Y, weights, metric=metric) == approx(
        weights @ distance(X, Q @ Y @ Q.T, metric=metric) ** 2
    )


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
def test_grassmann_grad(metric, get_mats, get_weights):
    """Test that gradient is the derivative of the loss"""
    n_matrices, n_channels = 5, 3
    X = get_mats(n_matrices, n_channels, "spd")
    Y = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices)
    Q = get_mats(1, n_channels, "orth")[0]

    eps = 1e-6
    grad_num = np.zeros((n_channels, n_channels))
    for i in range(n_channels):
        for j in range(n_channels):
            step = np.zeros((n_channels, n_channels))
            step[i, j] = eps
            grad_num[i, j] = (
                _loss(Q + step, X, Y, weights, metric=metric)
                - _loss(Q - step, X, Y, weights, metric=metric)
            ) / (2 * eps)

    grad = _grad(Q, X, Y, weights, metric=metric)
    assert grad.dtype == grad_num.dtype
    assert_array_almost_equal(grad, grad_num, decimal=5)


@pytest.mark.parametrize("metric", ["euclid", "riemann"])
def test_get_rotation_manifold(metric, get_mats, get_weights):
    n_matrices, n_channels = 5, 4
    X_source = get_mats(n_matrices, n_channels, "spd")
    X_target = get_mats(n_matrices, n_channels, "spd")
    weights = get_weights(n_matrices)

    Q = _get_rotation_manifold(X_source, X_target, weights, metric=metric)

    assert Q.shape == (n_channels, n_channels)
    assert _is_orth(Q)


@pytest.mark.parametrize("expl_var", [0.999, 4])
def test_get_rotation_tangentspace(rndstate, expl_var):
    """Test that Procrustes analysis maps source onto target"""
    n_vectors, n_ts = 20, 4
    X_source = rndstate.randn(n_vectors, n_ts)
    rotation = np.linalg.qr(rndstate.randn(n_ts, n_ts))[0]
    X_target = X_source @ rotation

    Q = _get_rotation_tangentspace(X_source, X_target, expl_var)
    assert _is_orth(Q)

    assert_array_almost_equal(Q, rotation)
    assert_array_almost_equal(X_source @ Q, X_target)
