import numpy as np
import pytest
from pytest import approx

from pyriemann.utils.geodesic import (
    geodesic,
    geodesic_euclid,
    geodesic_logeuclid,
    geodesic_riemann,
)
from pyriemann.utils.mean import (
    mean_euclid,
    mean_logeuclid,
    mean_riemann,
)


def get_geod_func():
    geod_func = [geodesic_euclid, geodesic_logeuclid, geodesic_riemann]
    for gf in geod_func:
        yield gf


def get_geod_name():
    geod_name = ["euclid", "logeuclid", "riemann"]
    for gn in geod_name:
        yield gn


@pytest.mark.parametrize(
    "geodesic_func", [geodesic_euclid, geodesic_logeuclid, geodesic_riemann]
)
class GeodesicFuncTestCase:
    def test_simple_mat(self, geodesic_func, get_covmats):
        n_channels = 3
        if geodesic_func is geodesic_euclid:
            A = 1.0 * np.eye(n_channels)
            B = 2.0 * np.eye(n_channels)
            Ctrue = 1.5 * np.eye(n_channels)
        else:
            A = 0.5 * np.eye(n_channels)
            B = 2 * np.eye(n_channels)
            Ctrue = np.eye(n_channels)
        self.geodesic_0(geodesic_func, A, B)
        self.geodesic_1(geodesic_func, A, B)
        self.geodesic_middle(geodesic_func, A, B, Ctrue)

    def test_random_mat(self, geodesic_func, get_covmats):
        n_matrices, n_channels = 2, 5
        covmats = get_covmats(n_matrices, n_channels)
        A, B = covmats[0], covmats[1]
        if geodesic_func is geodesic_euclid:
            Ctrue = mean_euclid(covmats)
        elif geodesic_func is geodesic_logeuclid:
            Ctrue = mean_logeuclid(covmats)
        elif geodesic_func is geodesic_riemann:
            Ctrue = mean_riemann(covmats)
        self.geodesic_0(geodesic_func, A, B)
        self.geodesic_1(geodesic_func, A, B)
        self.geodesic_middle(geodesic_func, A, B, Ctrue)


class TestGeodesicFunc(GeodesicFuncTestCase):
    def geodesic_0(self, geodesic_func, A, B):
        assert geodesic_func(A, B, 0) == approx(A)

    def geodesic_1(self, geodesic_func, A, B):
        assert geodesic_func(A, B, 1) == approx(B)

    def geodesic_middle(self, geodesic_func, A, B, Ctrue):
        assert geodesic_func(A, B, 0.5) == approx(Ctrue)


@pytest.mark.parametrize(
    "geodesic_func", [geodesic_euclid, geodesic_logeuclid, geodesic_riemann]
)
def test_geodesic_ndarray(geodesic_func, get_covmats):
    n_matrices, n_channels = 5, 3
    A = get_covmats(n_matrices, n_channels)
    B = get_covmats(n_matrices, n_channels)
    assert geodesic_func(A[0], B[0]).shape == A[0].shape  # 2D arrays
    assert geodesic_func(A, B).shape == A.shape  # 3D arrays

    n_sets = 5
    C = np.asarray([A for _ in range(n_sets)])
    D = np.asarray([B for _ in range(n_sets)])
    assert geodesic_func(C, D).shape == C.shape  # 4D arrays


@pytest.mark.parametrize("metric", get_geod_name())
def test_geodesic_wrapper_simple(metric):
    n_channels = 3
    if metric == "euclid":
        A = 1.0 * np.eye(n_channels)
        B = 2.0 * np.eye(n_channels)
        Ctrue = 1.5 * np.eye(n_channels)
    else:
        A = 0.5 * np.eye(n_channels)
        B = 2 * np.eye(n_channels)
        Ctrue = np.eye(n_channels)
    assert geodesic(A, B, 0.5, metric=metric) == approx(Ctrue)


@pytest.mark.parametrize("met, gfunc", zip(get_geod_name(), get_geod_func()))
def test_geodesic_wrapper_random(met, gfunc, get_covmats):
    n_matrices, n_channels = 2, 5
    covmats = get_covmats(n_matrices, n_channels)
    A, B = covmats[0], covmats[1]
    if gfunc is geodesic_euclid:
        Ctrue = mean_euclid(covmats)
    elif gfunc is geodesic_logeuclid:
        Ctrue = mean_logeuclid(covmats)
    elif gfunc is geodesic_riemann:
        Ctrue = mean_riemann(covmats)
    assert geodesic(A, B, 0.5, metric=met) == approx(Ctrue)
