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
    if gfun is geodesic_euclid:
        A = 1.0 * np.eye(n_channels)
        B = 2.0 * np.eye(n_channels)
        Ctrue = 1.5 * np.eye(n_channels)
    elif gfun is geodesic_wasserstein:
        A = 0.5 * np.eye(n_channels)
        B = 2 * np.eye(n_channels)
        Ctrue = 1.125 * np.eye(n_channels)
    else:
        A = 0.5 * np.eye(n_channels)
        B = 2 * np.eye(n_channels)
        Ctrue = np.eye(n_channels)
    assert_geodesic_0(gfun, A, B)
    assert_geodesic_1(gfun, A, B)
    assert_geodesic_middle(gfun, A, B, Ctrue)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("gfun", get_geod_func())
def test_geodesic_all_random(kind, gfun, get_mats):
    n_matrices, n_channels = 2, 5
    mats = get_mats(n_matrices, n_channels, kind)
    A, B = mats[0], mats[1]
    if gfun is geodesic_euclid:
        Ctrue = mean_euclid(mats)
    elif gfun is geodesic_logchol:
        Ctrue = mean_logchol(mats)
    elif gfun is geodesic_logeuclid:
        Ctrue = mean_logeuclid(mats)
    elif gfun is geodesic_riemann:
        Ctrue = mean_riemann(mats)
    elif gfun is geodesic_wasserstein:
        Ctrue = mean_wasserstein(mats)
    assert_geodesic_0(gfun, A, B)
    assert_geodesic_1(gfun, A, B)
    assert_geodesic_middle(gfun, A, B, Ctrue)


@pytest.mark.parametrize("complex_valued", [True, False])
def test_geodesic_euclid(rndstate, complex_valued):
    """Test Euclidean geodesic for generic matrices"""
    n_matrices, n_dim0, n_dim1 = 2, 3, 4
    mats = rndstate.randn(n_matrices, n_dim0, n_dim1)
    if complex_valued:
        mats = mats + 1j * rndstate.randn(n_matrices, n_dim0, n_dim1)
    A, B = mats[0], mats[1]
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
    A = 0.5 * np.eye(n_channels)
    if metric == "euclid":
        B = 1.5 * np.eye(n_channels)
    elif metric == "wasserstein":
        B = (9/2 - np.sqrt(8)) * np.eye(n_channels)
    else:
        B = 2.0 * np.eye(n_channels)
    assert geodesic(A, B, 0.5, metric=metric) == approx(np.eye(n_channels))


@pytest.mark.parametrize("metric, gfun", zip(get_geod_name(), get_geod_func()))
def test_geodesic_wrapper_random(metric, gfun, get_mats):
    n_matrices, n_channels = 2, 5
    mats = get_mats(n_matrices, n_channels, "spd")
    A, B = mats[0], mats[1]
    if gfun is geodesic_euclid:
        Ctrue = mean_euclid(mats)
    elif gfun is geodesic_logchol:
        Ctrue = mean_logchol(mats)
    elif gfun is geodesic_logeuclid:
        Ctrue = mean_logeuclid(mats)
    elif gfun is geodesic_riemann:
        Ctrue = mean_riemann(mats)
    elif gfun is geodesic_wasserstein:
        Ctrue = mean_wasserstein(mats)
    assert geodesic(A, B, 0.5, metric=metric) == approx(Ctrue)
