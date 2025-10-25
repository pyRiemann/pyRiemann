import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.geodesic import (
    geodesic,
    geodesic_euclid,
    geodesic_logchol,
    geodesic_logeuclid,
    geodesic_riemann,
    geodesic_wasserstein
)
from pyriemann.utils.mean import (
    mean_euclid,
    mean_logchol,
    mean_logeuclid,
    mean_riemann,
    mean_wasserstein,
)


def get_geod_func():
    geod_func = [
        geodesic_euclid,
        geodesic_logchol,
        geodesic_logeuclid,
        geodesic_riemann,
        geodesic_wasserstein
    ]
    for gf in geod_func:
        yield gf


def get_geod_name():
    geod_name = [
        "euclid",
        "logchol",
        "logeuclid",
        "riemann",
        "wasserstein"
    ]
    for gn in geod_name:
        yield gn


def assert_geodesic_0(gfun, A, B):
    assert gfun(A, B, 0) == approx(A)


def assert_geodesic_1(gfun, A, B):
    assert gfun(A, B, 1) == approx(B)


def assert_geodesic_middle(gfun, A, B, Ctrue):
    assert gfun(A, B, 0.5) == approx(Ctrue)


@pytest.mark.parametrize("gfun", get_geod_func())
def test_geodesic_all_simple(gfun):
    n_channels = 3
    eye = np.eye(n_channels)
    if gfun is geodesic_euclid:
        A = 1.0 * eye
        B = 2.0 * eye
        Ctrue = 1.5 * eye
    elif gfun is geodesic_wasserstein:
        A = 0.5 * eye
        B = 2 * eye
        Ctrue = 1.125 * eye
    else:
        A = 0.5 * eye
        B = 2 * eye
        Ctrue = eye
    assert_geodesic_0(gfun, A, B)
    assert_geodesic_1(gfun, A, B)
    assert_geodesic_middle(gfun, A, B, Ctrue)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("gfun", get_geod_func())
def test_geodesic_all_random(kind, gfun, get_mats):
    n_matrices, n_channels = 2, 5
    X = get_mats(n_matrices, n_channels, kind)
    A, B = X[0], X[1]
    if gfun is geodesic_euclid:
        Ctrue = mean_euclid(X)
    elif gfun is geodesic_logchol:
        Ctrue = mean_logchol(X)
    elif gfun is geodesic_logeuclid:
        Ctrue = mean_logeuclid(X)
    elif gfun is geodesic_riemann:
        Ctrue = mean_riemann(X)
    elif gfun is geodesic_wasserstein:
        Ctrue = mean_wasserstein(X)
    assert_geodesic_0(gfun, A, B)
    assert_geodesic_1(gfun, A, B)
    assert_geodesic_middle(gfun, A, B, Ctrue)


@pytest.mark.parametrize("kind", ["real", "comp"])
def test_geodesic_euclid(kind, get_mats):
    """Euclidean geodesic for non-square matrices"""
    n_dim1, n_dim2 = 3, 4
    A, B = get_mats(2, [n_dim1, n_dim2], kind)
    assert geodesic_euclid(A, B).shape == A.shape


@pytest.mark.parametrize("metric", get_geod_name())
def test_geodesic_wrapper_ndarray(metric, get_mats):
    n_matrices, n_channels = 5, 3
    A = get_mats(n_matrices, n_channels, "spd")
    B = get_mats(n_matrices, n_channels, "spd")
    assert geodesic(A[0], B[0], .3, metric=metric).shape == A[0].shape
    assert geodesic(A, B, .2, metric=metric).shape == A.shape  # 3D arrays

    n_sets = 4
    C = np.asarray([A for _ in range(n_sets)])
    D = np.asarray([B for _ in range(n_sets)])
    assert geodesic(C, D, .7, metric=metric).shape == C.shape  # 4D arrays


@pytest.mark.parametrize("metric", get_geod_name())
def test_geodesic_wrapper_simple(metric):
    n_channels = 3
    eye = np.eye(n_channels)
    A = 0.5 * eye
    if metric == "euclid":
        B = 1.5 * eye
    elif metric == "wasserstein":
        B = (9/2 - np.sqrt(8)) * eye
    else:
        B = 2.0 * eye
    assert geodesic(A, B, 0.5, metric=metric) == approx(eye)


@pytest.mark.parametrize("metric, gfun", zip(get_geod_name(), get_geod_func()))
def test_geodesic_wrapper_random(metric, gfun, get_mats):
    n_channels = 5
    X = get_mats(2, n_channels, "spd")
    if gfun is geodesic_euclid:
        Ctrue = mean_euclid(X)
    elif gfun is geodesic_logchol:
        Ctrue = mean_logchol(X)
    elif gfun is geodesic_logeuclid:
        Ctrue = mean_logeuclid(X)
    elif gfun is geodesic_riemann:
        Ctrue = mean_riemann(X)
    elif gfun is geodesic_wasserstein:
        Ctrue = mean_wasserstein(X)
    assert geodesic(X[0], X[1], 0.5, metric=metric) == approx(Ctrue)
