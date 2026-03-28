import numpy as np
from numpy.testing import assert_array_almost_equal
import pytest
from pytest import approx
from scipy.linalg import block_diag
from scipy.signal import welch, csd, coherence as coherence_sp
from sklearn.covariance import empirical_covariance

from pyriemann.utils.covariance import (
    covariances, covariance_scm, covariances_EP, covariances_X, eegtocov,
    cross_spectrum, cospectrum, coherence,
    normalize, get_nondiag_weight, block_covariances, _complex_estimator
)
from pyriemann.utils.test import (
    is_real,
    is_hermitian,
    is_sym_pos_def,
    is_herm_pos_def
)


estimators = ["corr", "cov", "lwf", "mcd", "oas", "sch", "scm"]
m_estimators = ["hub", "stu", "tyl"]


@pytest.mark.parametrize(
    "estimator", estimators + m_estimators + [np.cov, "truc", None]
)
def test_covariances(estimator, get_mats):
    """Test covariance for multiple estimators"""
    n_matrices, n_channels, n_times = 2, 3, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    if estimator is None:
        cov = covariances(X)
        assert cov.shape == (n_matrices, n_channels, n_channels)
    elif estimator == "truc":
        with pytest.raises(ValueError):
            covariances(X, estimator=estimator)
    else:
        cov = covariances(X, estimator=estimator)
        assert cov.shape == (n_matrices, n_channels, n_channels)
        assert is_sym_pos_def(cov)

        if estimator == "corr":
            assert_array_almost_equal(
                np.diagonal(cov, axis1=-2, axis2=-1),
                np.ones((n_matrices, n_channels))
            )
        elif estimator == "tyl":
            assert_array_almost_equal(
                np.trace(cov, axis1=-2, axis2=-1),
                np.full(n_matrices, n_channels)
            )


@pytest.mark.parametrize(
    "estimator",
    ["corr", "cov", "lwf", "oas", "sch", "scm", "hub", "stu", "tyl"]
)
def test_covariances_broadcasting(estimator, get_mats):
    n_dim5, n_dim4, n_matrices, n_channels, n_times = 2, 4, 2, 3, 100
    X = get_mats(n_dim5, [n_dim4, n_matrices, n_channels, n_times], "real")

    # 2D array
    C2 = covariances(X[0, 0, 0], estimator=estimator)
    assert C2.shape == (n_channels, n_channels)

    # 3D array
    C3 = covariances(X[0, 0], estimator=estimator)
    assert C3.shape == (n_matrices, n_channels, n_channels)
    assert C3[0] == approx(C2)

    # 4D array
    C4 = covariances(X[0], estimator=estimator)
    assert C4.shape == (n_dim4, n_matrices, n_channels, n_channels)
    assert C4[0, 0] == approx(C2)

    # 5D array
    C5 = covariances(X, estimator=estimator)
    assert C5.shape == (n_dim5, n_dim4, n_matrices, n_channels, n_channels)
    assert C5[0, 0, 0] == approx(C2)


@pytest.mark.parametrize("assume_centered", [True, False])
def test_covariance_scm_real(assume_centered, get_mats):
    """Test equivalence between pyriemann and sklearn estimator on real data"""
    n_matrices, n_channels, n_times = 3, 4, 50
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    cov = covariance_scm(X, assume_centered=assume_centered)
    cov_sklearn = np.asarray([
        empirical_covariance(x.T, assume_centered=assume_centered)
        for x in X
    ])
    assert_array_almost_equal(cov, cov_sklearn, 10)


def test_covariance_scm_complex(get_mats):
    """Test correctness of decorator for complex estimator on complex data"""
    n_matrices, n_channels, n_times = 4, 3, 60
    X = get_mats(n_matrices, [n_channels, n_times], "comp")

    cov = covariance_scm(X, assume_centered=True)

    @_complex_estimator
    def complex_scm_sklearn(x):
        return empirical_covariance(x.T, assume_centered=True)
    cov_decorator = np.asarray([complex_scm_sklearn(x) for x in X])
    assert_array_almost_equal(cov, cov_decorator, 10)


@pytest.mark.parametrize("kind", ["real", "comp"])
def test_covariance_scm_weights(kind, get_mats, get_weights):
    n_set, n_matrices, n_channels, n_times = 5, 4, 3, 20
    X = get_mats(n_set, [n_matrices, n_channels, n_times], kind)

    # test uniform weights
    cov = covariance_scm(X)
    covw = covariance_scm(X, weights=np.ones(n_times))
    assert cov == approx(covw)

    # setting one weight to almost 0 it's almost like not passing the sample
    weights = get_weights(n_times)
    cov = covariance_scm(X[..., 1:], weights=weights[1:])
    weights[0] = 1e-12
    covw = covariance_scm(X, weights=weights)
    assert cov == approx(covw)


@pytest.mark.parametrize("estimator", estimators + m_estimators)
def test_covariances_complex(estimator, get_mats):
    """Test covariance for complex inputs"""
    n_matrices, n_channels, n_times = 3, 4, 50
    X = get_mats(n_matrices, [n_channels, n_times], "comp")

    cov = covariances(X, estimator=estimator)
    assert cov.shape == (n_matrices, n_channels, n_channels)
    assert is_herm_pos_def(cov)


@pytest.mark.parametrize("kind", ["real", "comp"])
@pytest.mark.parametrize("estimator", estimators)
def test_covariances_ep(kind, estimator, get_mats):
    n_matrices, n_channels_x, n_channels_p, n_times = 2, 3, 3, 100
    X = get_mats(n_matrices, [n_channels_x, n_times], kind)
    P = get_mats(1, [n_channels_p, n_times], kind)[0]
    cov = covariances_EP(X, P, estimator=estimator)
    n_dim_cov = n_channels_x + n_channels_p
    assert cov.shape == (n_matrices, n_dim_cov, n_dim_cov)


@pytest.mark.parametrize(
    "estimator",
    ["corr", "cov", "lwf", "oas", "sch", "scm", "stu", "tyl"]
)
def test_covariances_ep_broadcasting(estimator, get_mats, rndstate):
    n_dim5, n_dim4, n_matrices, n_times = 3, 4, 2, 100
    n_channels_x, n_channels_p = 5, 6
    X = get_mats(n_dim5, [n_dim4, n_matrices, n_channels_x, n_times], "real")
    P = get_mats(1, [n_channels_p, n_times], "real")[0]

    # 2D array
    C2 = covariances_EP(X[0, 0, 0], P, estimator=estimator)
    n_dim_cov = n_channels_x + n_channels_p
    assert C2.shape == (n_dim_cov, n_dim_cov)

    # 3D array
    C3 = covariances_EP(X[0, 0], P, estimator=estimator)
    assert C3.shape == (n_matrices, n_dim_cov, n_dim_cov)
    assert C3[0] == approx(C2)

    # 4D array
    C4 = covariances_EP(X[0], P, estimator=estimator)
    assert C4.shape == (n_dim4, n_matrices, n_dim_cov, n_dim_cov)
    assert C4[0, 0] == approx(C2)

    # 5D array
    C5 = covariances_EP(X, P, estimator=estimator)
    assert C5.shape == (n_dim5, n_dim4, n_matrices, n_dim_cov, n_dim_cov)
    assert C5[0, 0, 0] == approx(C2)


@pytest.mark.parametrize("estimator", estimators)
def test_covariances_x(estimator, get_mats):
    if estimator == "mcd":
        return

    n_matrices, n_channels, n_times = 3, 5, 15
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    cov = covariances_X(X, estimator=estimator, alpha=5.)
    n_dim_cov = n_channels + n_times
    assert cov.shape == (n_matrices, n_dim_cov, n_dim_cov)

    # test broadcasting
    n_dim5, n_dim4, n_matrices, n_channels, n_times = 3, 4, 2, 5, 15
    X = get_mats(n_dim5, [n_dim4, n_matrices, n_channels, n_times], "real")

    # 2D array
    C2 = covariances_X(X[0, 0, 0], alpha=4., estimator=estimator)
    assert C2.shape == (n_dim_cov, n_dim_cov)

    # 5D array
    C5 = covariances_X(X, alpha=4., estimator=estimator)
    assert C5.shape == (n_dim5, n_dim4, n_matrices, n_dim_cov, n_dim_cov)
    assert C5[0, 0, 0] == approx(C2)


@pytest.mark.parametrize("estimator", estimators)
def test_block_covariances_est(estimator, get_mats):
    n_matrices, n_channels, n_times = 2, 12, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    cov = block_covariances(X, [4, 4, 4], estimator=estimator)
    assert cov.shape == (n_matrices, n_channels, n_channels)

    # test broadcasting
    if estimator == "mcd":
        return
    n_dim5, n_dim4, n_matrices, n_channels, n_times = 3, 2, 2, 8, 82
    X = get_mats(n_dim5, [n_dim4, n_matrices, n_channels, n_times], "real")

    # 2D array
    C2 = block_covariances(X[0, 0, 0], [2, 4, 2], estimator=estimator)
    assert C2.shape == (n_channels, n_channels)

    # 5D array
    C5 = block_covariances(X, [2, 4, 2], estimator=estimator)
    assert C5.shape == (n_dim5, n_dim4, n_matrices, n_channels, n_channels)
    assert C5[0, 0, 0] == approx(C2)


def test_block_covariances(get_mats):
    n_matrices, n_channels, n_times = 2, 12, 100
    X = get_mats(n_matrices, [n_channels, n_times], "real")

    cov = block_covariances(X, [12], estimator="cov")
    assert_array_almost_equal(cov, covariances(X, estimator="cov"))

    cov = block_covariances(X, [6, 6], estimator="cov")
    cov2 = covariances(X, estimator="cov")
    covcomp = block_diag(*(cov2[0, :6, :6], cov2[0, 6:12, 6:12]))
    assert_array_almost_equal(cov[0], covcomp)

    cov = block_covariances(X, [3, 5, 4], estimator="cov")
    cov2 = covariances(X, estimator="cov")
    covcomp = block_diag(*(cov2[0, :3, :3],
                           cov2[0, 3:8, 3:8],
                           cov2[0, 8:12, 8:12]))
    assert_array_almost_equal(cov[0], covcomp)


def test_eegtocov(rndstate):
    n_times, n_channels = 1000, 3
    X = rndstate.randn(n_times, n_channels)
    cov = eegtocov(X)
    assert cov.shape[1:] == (n_channels, n_channels)


def test_cross_spectrum_errors(rndstate):
    n_channels, n_times = 3, 1000
    X = rndstate.randn(n_channels, n_times)
    cross_spectrum(X)
    cross_spectrum(X, fs=128, fmin=2, fmax=40)
    cross_spectrum(X, fs=129, window=37)

    with pytest.raises(ValueError):  # window < 1
        cross_spectrum(X, window=0)
    with pytest.raises(ValueError):  # overlap <= 0
        cross_spectrum(X, overlap=0)
    with pytest.raises(ValueError):  # overlap >= 1
        cross_spectrum(X, overlap=1)
    with pytest.raises(ValueError):  # fmin > fmax
        cross_spectrum(X, fs=128, fmin=20, fmax=10)
    with pytest.raises(ValueError):  # fmax > fs/2
        cross_spectrum(X, fs=128, fmin=20, fmax=65)
    with pytest.warns(UserWarning):  # fs is None
        cross_spectrum(X, fmin=12)
    with pytest.warns(UserWarning):  # fs is None
        cross_spectrum(X, fmax=12)


def test_cross_spectrum(rndstate):
    n_channels, n_times = 3, 1000
    X = rndstate.randn(n_channels, n_times)

    c, freqs = cross_spectrum(X, fs=128, window=256)
    assert c.shape[0] == c.shape[1] == n_channels
    assert c.shape[-1] == freqs.shape[0]
    # test if cross-spectra are hermitian,
    # ie with symmetric real parts and skew-symmetric imag parts
    assert is_hermitian(np.transpose(c, (2, 0, 1)))
    # test if DC bins are real (always true)
    assert is_real(c[..., 0])
    # test if Nyquist bins are real (true when window is even)
    assert is_real(c[..., -1])
    # test if auto-spectra are real
    assert is_real(c.diagonal())


def test_cross_spectrum_scipy_auto(rndstate):
    """"Test equivalence between pyriemann and scipy for (auto-)spectra"""
    n_times = 1000
    X = rndstate.randn(5, n_times)
    fs, window, overlap = 128, 256, 0.75

    spect_pr, freqs_pr = cross_spectrum(
        X,
        fs=fs,
        window=window,
        overlap=overlap,
    )
    spect_pr = np.diagonal(spect_pr.real).T  # auto-spectra on diagonal
    spect_pr = spect_pr / np.linalg.norm(spect_pr)  # unit norm

    freqs_sp, spect_sp = welch(
        X,
        fs=fs,
        nperseg=window,
        noverlap=int(overlap * window),
        window=np.hanning(window),
        detrend=False,
        scaling="spectrum",
    )
    spect_sp /= np.linalg.norm(spect_sp)  # unit norm

    # compare frequencies
    assert_array_almost_equal(freqs_pr, freqs_sp, 6)
    # compare auto-spectra
    assert_array_almost_equal(spect_pr, spect_sp, 6)


def test_cross_spectrum_scipy_cross(rndstate):
    """"Test equivalence between pyriemann and scipy for cross-spectra"""
    n_times = 1000
    X = rndstate.randn(2, n_times)
    fs, window, overlap = 64, 128, 0.5

    cross_pr, freqs_pr = cross_spectrum(
        X,
        fs=fs,
        window=window,
        overlap=overlap)
    cross_pr = cross_pr[0, 1] / np.linalg.norm(cross_pr[0, 1])  # unit norm

    freqs_sp, cross_sp = csd(
        X[0],
        X[1],
        fs=fs,
        nperseg=window,
        noverlap=int(overlap * window),
        window=np.hanning(window),
        detrend=False,
        scaling="spectrum",
    )
    cross_sp /= np.linalg.norm(cross_sp)  # unit norm

    # compare frequencies
    assert_array_almost_equal(freqs_pr, freqs_sp, 6)
    # compare cross-spectra
    assert_array_almost_equal(cross_pr, cross_sp, 6)


def test_cross_spectrum_broadcasting(rndstate):
    n_dim4, n_matrices, n_channels, n_times = 4, 6, 3, 1000
    X = rndstate.randn(n_dim4, n_matrices, n_channels, n_times)

    # 2D array
    window = 64
    C2, _ = cross_spectrum(X[0, 0], window=window)
    n_freqs = window // 2 + 1
    assert C2.shape == (n_channels, n_channels, n_freqs)

    # 4D array
    C4, _ = cross_spectrum(X, window=window)
    assert C4.shape == (n_dim4, n_matrices, n_channels, n_channels, n_freqs)
    assert C4[0, 0] == approx(C2)


def test_cospectrum(rndstate):
    X = rndstate.randn(3, 1000)
    cospectrum(X)
    cospectrum(X, fs=128, fmin=2, fmax=40)

    # test equivalence between pyriemann and scipy for cospectra
    fs, window, overlap = 128, 256, 0.75
    cosp_pr, freqs_pr = cospectrum(X, fs=fs, window=window, overlap=overlap)
    cosp_pr = cosp_pr[0, 1] / np.linalg.norm(cosp_pr[0, 1])  # unit norm
    freqs_sp, cross_sp = csd(
        X[0],
        X[1],
        fs=fs,
        nperseg=window,
        noverlap=int(overlap * window),
        window=np.hanning(window),
        detrend=False,
        scaling="spectrum",
    )
    cosp_sp = cross_sp.real / np.linalg.norm(cross_sp.real)  # unit norm
    # compare frequencies
    assert_array_almost_equal(freqs_pr, freqs_sp, 6)
    # compare co-spectra
    assert_array_almost_equal(cosp_pr, cosp_sp, 6)


def test_cospectrum_broadcasting(rndstate):
    n_dim4, n_matrices, n_channels, n_times = 5, 4, 3, 100
    X = rndstate.randn(n_dim4, n_matrices, n_channels, n_times)

    # 2D array
    window = 64
    C2, _ = cospectrum(X[0, 0], window=window)
    n_freqs = window // 2 + 1
    assert C2.shape == (n_channels, n_channels, n_freqs)

    # 4D array
    C4, _ = cospectrum(X, window=window)
    assert C4.shape == (n_dim4, n_matrices, n_channels, n_channels, n_freqs)
    assert C4[0, 0] == approx(C2)


@pytest.mark.parametrize(
    "coh", ["ordinary", "instantaneous", "lagged", "imaginary"]
)
def test_coherence(coh, rndstate):
    n_channels, n_times = 3, 2048
    X = rndstate.randn(n_channels, n_times)

    c, freqs = coherence(X, fs=128, fmin=3, fmax=40, coh=coh)
    assert c.shape[0] == c.shape[1] == n_channels
    assert c.shape[-1] == freqs.shape[0]
    # test if coherence in [0,1]
    assert np.all((0. <= c) & (c <= 1.))

    # test equivalence between pyriemann and scipy for ordinary coherence
    if coh == "ordinary":
        fs, window, overlap = 128, 256, 0.75
        coh_pr, freqs_pr = coherence(X, fs=fs, window=window, overlap=overlap)
        freqs_sp, coh_sp = coherence_sp(
            X[0],
            X[1],
            fs=fs,
            nperseg=window,
            noverlap=int(overlap * window),
            window=np.hanning(window),
            detrend=False,
        )
        # compare frequencies
        assert_array_almost_equal(freqs_pr, freqs_sp, 6)
        # compare coherence
        assert_array_almost_equal(coh_pr[0, 1], coh_sp, 6)

    if coh == "lagged":
        with pytest.warns(UserWarning):  # not defined for DC and Nyquist bins
            coherence(X, coh=coh)
        with pytest.warns(UserWarning):  # not defined for DC and Nyquist bins
            coherence(X, fs=64, coh=coh)
    else:
        coherence(X, coh=coh)
        coherence(X, fs=64, coh=coh)


@pytest.mark.parametrize(
    "coh", ["ordinary", "instantaneous", "lagged", "imaginary"]
)
def test_coherence_properties(coh, rndstate):
    """Test statistical properties of coherence btw phase shifted channels"""
    fs, ft, n_periods = 16, 4, 20
    t = np.arange(0, n_periods, 1 / fs)
    n_times = t.shape[0]

    X, noise = np.empty((4, len(t))), 1e-9
    # reference channel: a pure sine + small noise (to avoid nan or inf)
    X[0] = np.sin(2 * np.pi * ft * t) + noise * rndstate.randn((n_times))
    # pi/4 shifted channel = pi/4 lagged phase
    X[1] = np.sin(2 * np.pi * ft * t + np.pi / 4) \
        + noise * rndstate.randn((n_times))
    # pi/2 shifted channel = quadrature phase
    X[2] = np.sin(2 * np.pi * ft * t + np.pi / 2) \
        + noise * rndstate.randn((n_times))
    # pi shifted channel = opposite phase
    X[3] = np.sin(2 * np.pi * ft * t + np.pi) \
        + noise * rndstate.randn((n_times))

    c, freqs = coherence(X, fs=fs, fmin=1, fmax=fs/2-1, window=fs,
                         overlap=0.5, coh=coh)
    foi = (freqs == ft)

    if coh == "ordinary":
        # ord coh equal 1 between ref and all other channels
        assert_array_almost_equal(c[..., foi], np.ones_like(c[..., foi]))

    elif coh == "instantaneous":
        # inst coh equal 0.5 between ref and pi/4 lagged phase channels
        assert c[0, 1, foi] == pytest.approx(0.5)
        # inst coh equal 0 between ref and quadrature phase channels
        assert c[0, 2, foi] == pytest.approx(0.0)
        # inst coh equal 1 between ref and opposite phase channels
        assert c[0, 3, foi] == pytest.approx(1.0)

    elif coh == "lagged":
        # lagged coh equal 1 between ref and quadrature phase channels
        assert c[0, 2, foi] == pytest.approx(1.0)
        # lagged coh equal 0 between ref and opposite phase channels
        assert c[0, 3, foi] == pytest.approx(0.0, abs=1e-4)

    elif coh == "imaginary":
        # imag coh equal 0.5 between ref and pi/4 lagged phase channels
        assert c[0, 1, foi] == pytest.approx(0.5)
        # imag coh equal 1 between ref and quadrature phase channels
        assert c[0, 2, foi] == pytest.approx(1.0)
        # imag coh equal 0 between ref and opposite phase channels
        assert c[0, 3, foi] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "coh", ["ordinary", "instantaneous", "lagged", "imaginary"]
)
def test_coherence_broadcasting(coh, rndstate):
    n_dim4, n_matrices, n_channels, n_times = 5, 4, 3, 100
    X = rndstate.randn(n_dim4, n_matrices, n_channels, n_times)

    # 2D array
    window = 64
    C2, _ = coherence(X[0, 0], window=window, coh=coh)
    n_freqs = window // 2 + 1
    assert C2.shape == (n_channels, n_channels, n_freqs)

    # 4D array
    C4, _ = coherence(X, window=window, coh=coh)
    assert C4.shape == (n_dim4, n_matrices, n_channels, n_channels, n_freqs)
    assert C4[0, 0] == approx(C2)


def test_coherence_error(rndstate):
    n_channels, n_times = 3, 50
    X = rndstate.randn(n_channels, n_times)
    with pytest.raises(ValueError):  # unknown coh
        coherence(X, coh="foobar")


@pytest.mark.parametrize("norm", ["corr", "trace", "determinant"])
def test_normalize_broadcasting(norm, rndstate):
    n_dim5, n_dim4, n_matrices, n_channels = 2, 6, 5, 3
    X = rndstate.randn(n_dim5, n_dim4, n_matrices, n_channels, n_channels)

    # 2D array
    N2 = normalize(X[0, 0, 0], norm)
    assert N2.shape == (n_channels, n_channels)

    # 3D array
    N3 = normalize(X[0, 0], norm)
    assert N3.shape == (n_matrices, n_channels, n_channels)
    assert N3[0] == approx(N2)

    # 4D array
    N4 = normalize(X[0], norm)
    assert N4.shape == (n_dim4, n_matrices, n_channels, n_channels)
    assert N4[0, 0] == approx(N2)

    # 5D array
    N5 = normalize(X, norm)
    assert N5.shape == (n_dim5, n_dim4, n_matrices, n_channels, n_channels)
    assert N5[0, 0, 0] == approx(N2)


def test_normalize(rndstate, get_mats):
    n_matrices, n_channels = 20, 3

    # after corr-normalization => diags = 1 and values in [-1, 1]
    X = get_mats(n_channels, n_channels, "spd")
    Xcn = normalize(X, "corr")
    assert_array_almost_equal(np.ones(Xcn.shape[:-1]),
                              np.diagonal(Xcn, axis1=-2, axis2=-1))
    assert np.all(-1 <= Xcn) and np.all(Xcn <= 1)

    # after trace-normalization => trace equal to 1
    X = rndstate.randn(n_matrices, n_channels, n_channels)
    Xtn = normalize(X, "trace")
    assert_array_almost_equal(np.ones(Xtn.shape[0]),
                              np.trace(Xtn, axis1=-2, axis2=-1))

    # after determinant-normalization => determinant equal to +/- 1
    Xdn = normalize(X, "determinant")
    assert_array_almost_equal(np.ones(Xdn.shape[0]),
                              np.abs(np.linalg.det(Xdn)))


def test_normalize_errors(rndstate):
    n_matrices, n_channels = 3, 2
    with pytest.raises(ValueError):  # not at least 2d
        normalize(rndstate.randn(n_channels), "trace")
    with pytest.raises(ValueError):  # not square
        shape = (n_matrices, n_channels, n_channels + 2)
        normalize(rndstate.randn(*shape), "trace")
    with pytest.raises(ValueError):  # invalid normalization type
        normalize(rndstate.randn(n_matrices, n_channels, n_channels), "abc")


def test_get_nondiag_weight_broadcasting(rndstate):
    n_dim5, n_dim4, n_matrices, n_channels = 2, 6, 5, 3
    X = rndstate.randn(n_dim5, n_dim4, n_matrices, n_channels, n_channels)

    # 2D array
    w2 = get_nondiag_weight(X[0, 0, 0])
    assert np.isscalar(w2)

    # 3D array
    W3 = get_nondiag_weight(X[0, 0])
    assert W3.shape == (n_matrices,)
    assert W3[0] == approx(w2)

    # 4D array
    W4 = get_nondiag_weight(X[0])
    assert W4.shape == (n_dim4, n_matrices,)
    assert W4[0, 0] == approx(w2)

    # 5D array
    W5 = get_nondiag_weight(X)
    assert W5.shape == (n_dim5, n_dim4, n_matrices,)
    assert W5[0, 0, 0] == approx(w2)


def test_get_nondiag_weight(rndstate):
    n_matrices, n_channels = 7, 3

    # 2x2 constant matrices => non-diag weights equal to 1
    X = rndstate.randn(n_matrices, 1, 1) * np.ones((n_matrices, 2, 2))
    w = get_nondiag_weight(X)
    assert_array_almost_equal(w, np.ones(n_matrices))

    # diagonal matrices => non-diag weights equal to 0
    X = rndstate.randn(n_matrices, 1, 1) * ([np.eye(n_channels)] * n_matrices)
    w = get_nondiag_weight(X)
    assert_array_almost_equal(w, np.zeros(n_matrices))


def test_get_nondiag_weight_errors(rndstate):
    n_matrices, n_channels = 3, 2
    with pytest.raises(ValueError):  # not at least 2d
        get_nondiag_weight(rndstate.randn(n_channels))
    with pytest.raises(ValueError):  # not square
        shape = (n_matrices, n_channels, n_channels + 2)
        get_nondiag_weight(rndstate.randn(*shape))
