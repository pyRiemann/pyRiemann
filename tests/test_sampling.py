import numpy as np
from numpy.testing import assert_array_equal
import pytest

from pyriemann.datasets import sample_gaussian, RandomOverSampler
from pyriemann.geometry.distance import distance_riemann
from pyriemann.geometry.test import is_herm_pos_def as is_hpd

pytestmark = pytest.mark.numpy_only


@pytest.mark.parametrize("dtype", [np.float64, np.complex64])
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("sampling_method", ["auto", "slice", "rejection"])
@pytest.mark.parametrize("sigma", [1.1, 3])
def test_sample_gaussian_floatsigma_d2(dtype, n_jobs, sampling_method, sigma):
    """Test for dim=2 with a float sigma."""
    n_matrices, n_dim = 3, 2
    mean = np.eye(n_dim, dtype=dtype)
    X = sample_gaussian(n_matrices, mean, sigma, random_state=42,
                        n_jobs=n_jobs, sampling_method=sampling_method)
    assert X.shape == (n_matrices, n_dim, n_dim)
    assert is_hpd(X)
    if dtype == np.complex64:
        assert np.iscomplexobj(X)


@pytest.mark.parametrize("dtype", [np.float64, np.complex64])
@pytest.mark.parametrize("n_dim", [3, 4])
@pytest.mark.parametrize("n_jobs", [1, -1])
@pytest.mark.parametrize("sampling_method", ["auto", "slice"])
def test_sample_gaussian_floatsigma(dtype, n_dim, n_jobs, sampling_method):
    """Test for dim>2 with a float sigma."""
    n_matrices = 3
    mean, sigma = np.eye(n_dim, dtype=dtype), 2.5
    X = sample_gaussian(n_matrices, mean, sigma, random_state=42,
                        n_jobs=n_jobs, sampling_method=sampling_method)
    assert X.shape == (n_matrices, n_dim, n_dim)
    assert is_hpd(X)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("n_dim", [3, 4])
def test_sample_gaussian_ndarraysigma(kind, n_dim, get_mats):
    """Test with a ndarray sigma."""
    n_matrices, n_ts = 5, n_dim * (n_dim + 1) // 2
    mean = get_mats(1, n_dim, kind)[0]
    sigma = np.eye(n_ts)
    X = sample_gaussian(n_matrices, mean, sigma, random_state=42)
    assert X.shape == (n_matrices, n_dim, n_dim)
    assert is_hpd(X)


def test_sample_gaussian_errors_samplingmethod():
    with pytest.raises(ValueError):  # unknown sampling method
        sample_gaussian(5, np.eye(2), 1., sampling_method="blabla")
    with pytest.raises(ValueError):  # dim=3 not yet supported with rejection
        n_dim = 3
        sample_gaussian(5, np.eye(n_dim), 1., sampling_method="rejection")


@pytest.mark.parametrize("dtype", [np.float64, np.complex64])
@pytest.mark.parametrize("n_jobs", [1, -1])
def test_sample_gaussian_dispersion(dtype, n_jobs):
    """Test dispersions of matrices sampled from Riemannian Gaussians"""
    n_matrices, n_dim, sig_1, sig_2 = 5, 4, 1., 2.
    mean = np.eye(n_dim, dtype=dtype)
    X1 = sample_gaussian(
        n_matrices, mean, sig_1, random_state=42, n_jobs=n_jobs
    )
    X2 = sample_gaussian(
        n_matrices, mean, sig_2, random_state=66, n_jobs=n_jobs
    )
    avg_d1 = np.mean(distance_riemann(X1, mean))
    avg_d2 = np.mean(distance_riemann(X2, mean))
    assert avg_d1 < avg_d2


def test_sample_gaussian_errors_sigma():
    n_matrices, n_dim = 3, 4
    mean, sigma = np.eye(n_dim), 2.
    with pytest.raises(ValueError):  # mean is not a matrix
        sample_gaussian(n_matrices, np.ones(n_dim), sigma)
    with pytest.raises(ValueError):  # sigma is not the right shape
        sample_gaussian(n_matrices, mean, np.ones(n_dim))
    with pytest.raises(ValueError):  # n_matrices is negative
        sample_gaussian(-n_matrices, mean, sigma)
    with pytest.raises(ValueError):  # n_matrices is not an integer
        sample_gaussian(4.2, mean, sigma)


@pytest.mark.parametrize("kind", ["spd", "hpd"])
@pytest.mark.parametrize("metric", ["euclid", "logeuclid", "riemann"])
@pytest.mark.parametrize("n_jobs", [1, -1])
def test_random_over_sampler(kind, metric, n_jobs, get_mats):
    n_matrices_by_class, n_dim = 6, 2
    X = get_mats(2 * n_matrices_by_class, n_dim, kind)
    y = np.array([0] * n_matrices_by_class + [1] * n_matrices_by_class)

    ros = RandomOverSampler(metric=metric, n_jobs=n_jobs)
    ros.fit(X, y)
    Xr, yr = ros.fit_resample(X, y)

    assert Xr.ndim == 3
    assert Xr.shape[0] == yr.shape[0]
    assert Xr.shape[1:] == (n_dim, n_dim)


@pytest.mark.parametrize(
    "sampling_strategy",
    ["minority", "not minority", "not majority", "all", "auto"]
)
def test_random_over_sampler_strategy(sampling_strategy, get_mats):
    n_dim = 3
    n1, lab1 = 15, 0
    X1, y1 = get_mats(n1, n_dim, "spd"), np.full(n1, lab1)
    n2, lab2 = 10, 1
    X2, y2 = get_mats(n2, n_dim, "spd"), np.full(n2, lab2)
    n3, lab3 = 5, 2
    X3, y3 = get_mats(n3, n_dim, "spd"), np.full(n3, lab3)
    X, y = np.concatenate((X1, X2, X3)), np.concatenate((y1, y2, y3))

    ros = RandomOverSampler(sampling_strategy=sampling_strategy)
    Xr, yr = ros.fit_resample(X, y)

    assert_array_equal(X, Xr[:len(X)])
    assert_array_equal(y, yr[:len(y)])

    assert len(yr[yr == lab1]) == n1
    if sampling_strategy == "minority":
        assert len(yr[yr == lab2]) == n2
        assert len(yr[yr == lab3]) == n1
    elif sampling_strategy == "not minority":
        assert len(yr[yr == lab2]) == n1
        assert len(yr[yr == lab3]) == n3
    elif sampling_strategy in ["not majority", "auto", "all"]:
        assert len(yr[yr == lab2]) == n1
        assert len(yr[yr == lab3]) == n1
