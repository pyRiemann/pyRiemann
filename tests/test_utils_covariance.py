from numpy.testing import assert_array_almost_equal, assert_array_equal
import numpy as np
from scipy.signal import welch, csd, coherence as coherence_sp
import pytest

from pyriemann.utils.covariance import (covariances, covariances_EP, eegtocov,
                                        cross_spectrum, cospectrum, coherence,
                                        normalize, get_nondiag_weight)


@pytest.mark.parametrize(
    'estimator', ['oas', 'lwf', 'scm', 'corr', 'mcd', np.cov, 'truc', None]
)
def test_covariances(estimator):
    """Test covariance for multiple estimator"""
    x = np.random.randn(2, 3, 100)
    if estimator is None:
        cov = covariances(x)
        assert cov.shape == (2, 3, 3)
    elif estimator == 'truc':
        with pytest.raises(ValueError):
            covariances(x, estimator=estimator)
    else:
        cov = covariances(x, estimator=estimator)
        assert cov.shape == (2, 3, 3)


@pytest.mark.parametrize(
    'estimator', ['oas', 'lwf', 'scm', 'corr', 'mcd', None]
)
def test_covariances_EP(estimator):
    """Test covariance_EP for multiple estimator"""
    x = np.random.randn(2, 3, 100)
    p = np.random.randn(3, 100)

    if estimator is None:
        cov = covariances_EP(x, p)
    else:
        cov = covariances_EP(x, p, estimator=estimator)
    assert cov.shape == (2, 6, 6)


def test_covariances_eegtocov():
    """Test eegtocov"""
    x = np.random.randn(1000, 3)
    cov = eegtocov(x)
    assert cov.shape[1] == 3


def test_covariances_cross_spectrum():
    rs = np.random.RandomState(42)
    n_channels, n_times = 3, 1000
    x = rs.randn(n_channels, n_times)
    cross_spectrum(x)
    cross_spectrum(x, fs=128, fmin=2, fmax=40)

    with pytest.raises(ValueError):  # window < 1
        cross_spectrum(x, window=0)
    with pytest.raises(ValueError):  # overlap <= 0
        cross_spectrum(x, overlap=0)
    with pytest.raises(ValueError):  # overlap >= 1
        cross_spectrum(x, overlap=1)
    with pytest.raises(ValueError):  # fmin > fmax
        cross_spectrum(x, fs=128, fmin=20, fmax=10)
    with pytest.raises(ValueError):  # fmax > fs/2
        cross_spectrum(x, fs=128, fmin=20, fmax=65)
    with pytest.warns(UserWarning):  # fs is None
        cross_spectrum(x, fmin=12)
    with pytest.warns(UserWarning):  # fs is None
        cross_spectrum(x, fmax=12)

    c, freqs = cross_spectrum(x, fs=128, window=256)
    assert c.shape[0] == c.shape[1] == n_channels
    assert c.shape[-1] == freqs.shape[0]
    # test if real parts are symmetric
    assert_array_almost_equal(c.real, np.transpose(c.real, (1, 0, 2)), 6)
    # test if imag parts are skew-symmetric
    assert_array_almost_equal(c.imag, -np.transpose(c.imag, (1, 0, 2)), 6)
    # test if DC bins are real (always true)
    assert_array_almost_equal(c[..., 0].imag, np.zeros_like(c[..., 0].imag))
    # test if Nyquist bins are real (true when window is even)
    assert_array_almost_equal(c[..., -1].imag, np.zeros_like(c[..., -1].imag))
    # test if auto-spectra are real
    assert_array_almost_equal(c.imag.diagonal(),
                              np.zeros_like(c.imag.diagonal()))

    # test equivalence between pyriemann and scipy for (auto-)spectra
    x = rs.randn(5, n_times)
    fs, window, overlap = 128, 256, 0.75
    spect_pr, freqs_pr = cross_spectrum(
        x,
        fs=fs,
        window=window,
        overlap=overlap)
    spect_pr = np.diagonal(spect_pr.real).T  # auto-spectra on diagonal
    spect_pr = spect_pr / np.linalg.norm(spect_pr)  # unit norm
    freqs_sp, spect_sp = welch(
        x,
        fs=fs,
        nperseg=window,
        noverlap=int(overlap * window),
        window=np.hanning(window),
        detrend=False,
        scaling='spectrum')
    spect_sp /= np.linalg.norm(spect_sp)  # unit norm
    # compare frequencies
    assert_array_almost_equal(freqs_pr, freqs_sp, 6)
    # compare auto-spectra
    assert_array_almost_equal(spect_pr, spect_sp, 6)

    # test equivalence between pyriemann and scipy for cross-spectra
    x = rs.randn(2, n_times)
    fs, window, overlap = 64, 128, 0.5
    cross_pr, freqs_pr = cross_spectrum(
        x,
        fs=fs,
        window=window,
        overlap=overlap)
    cross_pr = cross_pr[0, 1] / np.linalg.norm(cross_pr[0, 1])  # unit norm
    freqs_sp, cross_sp = csd(
        x[0],
        x[1],
        fs=fs,
        nperseg=window,
        noverlap=int(overlap * window),
        window=np.hanning(window),
        detrend=False,
        scaling='spectrum')
    cross_sp /= np.linalg.norm(cross_sp)  # unit norm
    # compare frequencies
    assert_array_almost_equal(freqs_pr, freqs_sp, 6)
    # compare cross-spectra
    assert_array_almost_equal(cross_pr, cross_sp, 6)


def test_covariances_cospectrum():
    """Test cospectrum"""
    rs = np.random.RandomState(42)
    x = rs.randn(3, 1000)
    cospectrum(x)
    cospectrum(x, fs=128, fmin=2, fmax=40)

    # test equivalence between pyriemann and scipy for cospectra
    fs, window, overlap = 128, 256, 0.75
    cosp_pr, freqs_pr = cospectrum(x, fs=fs, window=window, overlap=overlap)
    cosp_pr = cosp_pr[0, 1] / np.linalg.norm(cosp_pr[0, 1])  # unit norm
    freqs_sp, cross_sp = csd(
        x[0],
        x[1],
        fs=fs,
        nperseg=window,
        noverlap=int(overlap * window),
        window=np.hanning(window),
        detrend=False,
        scaling='spectrum')
    cosp_sp = cross_sp.real / np.linalg.norm(cross_sp.real)  # unit norm
    # compare frequencies
    assert_array_almost_equal(freqs_pr, freqs_sp, 6)
    # compare co-spectra
    assert_array_almost_equal(cosp_pr, cosp_sp, 6)


@pytest.mark.parametrize(
    'coh', ['ordinary', 'instantaneous', 'lagged', 'imaginary', 'foobar']
)
def test_covariances_coherence(coh):
    """Test coherence"""
    rs = np.random.RandomState(42)
    n_channels, n_times = 3, 2048
    x = rs.randn(n_channels, n_times)

    if coh == 'foobar':
        with pytest.raises(ValueError):  # unknown coh
            coherence(x, coh=coh)
    else:
        coherence(x, coh=coh)
        coherence(x, fs=128, fmin=2, fmax=40, coh=coh)
        c, freqs = coherence(x, fs=32, window=256, coh=coh)
        assert c.shape[0] == c.shape[1] == n_channels
        assert c.shape[-1] == freqs.shape[0]
        # test if coherence in [0,1]
        assert np.all((0. <= c) & (c <= 1.))

    # test equivalence between pyriemann and scipy for ordinary coherence
    if coh == 'ordinary':
        fs, window, overlap = 128, 256, 0.75
        coh_pr, freqs_pr = coherence(x, fs=fs, window=window, overlap=overlap)
        freqs_sp, coh_sp = coherence_sp(
            x[0],
            x[1],
            fs=fs,
            nperseg=window,
            noverlap=int(overlap * window),
            window=np.hanning(window),
            detrend=False)
        # compare frequencies
        assert_array_almost_equal(freqs_pr, freqs_sp, 6)
        # compare coherence
        assert_array_almost_equal(coh_pr[0, 1], coh_sp, 6)

    # test statistical properties of coherence between phase shifted channels
    if coh in ['ordinary', 'instantaneous', 'lagged', 'imaginary']:
        fs, ft, n_periods = 16, 4, 20
        t = np.arange(0, n_periods, 1 / fs)
        n_times = t.shape[0]

        x = np.zeros((4, len(t)))
        noise = 1e-10
        # reference channel: a pure sine + small noise (to avoid nan or inf)
        x[0] = np.sin(2 * np.pi * ft * t) + noise * rs.randn((n_times))
        # pi/4 shifted channel = pi/4 lagged phase
        x[1] = np.sin(2 * np.pi * ft * t + np.pi / 4) \
            + noise * rs.randn((n_times))
        # pi/2 shifted channel = quadrature phase
        x[2] = np.sin(2 * np.pi * ft * t + np.pi / 2) \
            + noise * rs.randn((n_times))
        # pi shifted channel = opposite phase
        x[3] = np.sin(2 * np.pi * ft * t + np.pi) + noise * rs.randn((n_times))

        c, freqs = coherence(x, fs=fs, window=fs, overlap=0.5, coh=coh)
        foi = (freqs == ft)

        if coh == 'ordinary':
            # ord coh equal 1 between ref and all other channels
            assert_array_almost_equal(c[..., foi], np.ones_like(c[..., foi]))

        elif coh == 'instantaneous':
            # inst coh equal 0.5 between ref and pi/4 lagged phase channels
            assert c[0, 1, foi] == pytest.approx(0.5)
            # inst coh equal 0 between ref and quadrature phase channels
            assert c[0, 2, foi] == pytest.approx(0.0)
            # inst coh equal 1 between ref and opposite phase channels
            assert c[0, 3, foi] == pytest.approx(1.0)

        elif coh == 'lagged':
            # lagged coh equal 1 between ref and quadrature phase channels
            assert c[0, 2, foi] == pytest.approx(1.0)
            # lagged coh equal 0 between ref and opposite phase channels
            assert c[0, 3, foi] == pytest.approx(0.0, abs=1e-6)

        elif coh == 'imaginary':
            # imag coh equal 0.5 between ref and pi/4 lagged phase channels
            assert c[0, 1, foi] == pytest.approx(0.5)
            # imag coh equal 1 between ref and quadrature phase channels
            assert c[0, 2, foi] == pytest.approx(1.0)
            # imag coh equal 0 between ref and opposite phase channels
            assert c[0, 3, foi] == pytest.approx(0.0)


def test_normalize():
    """Test normalize"""
    rs = np.random.RandomState(42)
    n_conds, n_trials, n_channels = 15, 20, 3

    # test a 2d array, ie a single square matrix
    mat = rs.randn(n_channels, n_channels)
    mat_n = normalize(mat, "trace")
    assert_array_equal(mat.shape, mat_n.shape)
    # test a 3d array, ie a group of square matrices
    mat = rs.randn(n_trials, n_channels, n_channels)
    mat_n = normalize(mat, "determinant")
    assert_array_equal(mat.shape, mat_n.shape)
    # test a 4d array, ie a group of groups of square matrices
    mat = rs.randn(n_conds, n_trials, n_channels, n_channels)
    mat_n = normalize(mat, "trace")
    assert_array_equal(mat.shape, mat_n.shape)

    # after trace-normalization => trace equal to 1
    mat = rs.randn(n_trials, n_channels, n_channels)
    mat_tn = normalize(mat, "trace")
    assert_array_almost_equal(np.ones(mat_tn.shape[0]),
                              np.trace(mat_tn, axis1=-2, axis2=-1))
    # after determinant-normalization => determinant equal to +/- 1
    mat_dn = normalize(mat, "determinant")
    assert_array_almost_equal(np.ones(mat_dn.shape[0]),
                              np.abs(np.linalg.det(mat_dn)))

    with pytest.raises(ValueError):  # not at least 2d
        normalize(rs.randn(n_channels), "trace")
    with pytest.raises(ValueError):  # not square
        normalize(rs.randn(n_trials, n_channels, n_channels + 2), "trace")
    with pytest.raises(ValueError):  # invalid normalization type
        normalize(rs.randn(n_trials, n_channels, n_channels), "abc")


def test_get_nondiag_weight():
    """Test get_nondiag_weight"""
    rs = np.random.RandomState(17)
    n_conds, n_trials, n_channels = 10, 20, 3

    # test a 2d array, ie a single square matrix
    w = get_nondiag_weight(rs.randn(n_channels, n_channels))
    assert np.isscalar(w)
    # test a 3d array, ie a group of square matrices
    w = get_nondiag_weight(rs.randn(n_trials, n_channels, n_channels))
    assert_array_equal(w.shape, [n_trials])
    # test a 4d array, ie a group of groups of square matrices
    w = get_nondiag_weight(rs.randn(n_conds, n_trials, n_channels, n_channels))
    assert_array_equal(w.shape, [n_conds, n_trials])

    # 2x2 constant matrices => non-diag weights equal to 1
    mats = rs.randn(n_trials, 1, 1) * np.ones((n_trials, 2, 2))
    w = get_nondiag_weight(mats)
    assert_array_almost_equal(w, np.ones(n_trials))
    # diagonal matrices => non-diag weights equal to 0
    mats = rs.randn(n_trials, 1, 1) * ([np.eye(n_channels)] * n_trials)
    w = get_nondiag_weight(mats)
    assert_array_almost_equal(w, np.zeros(n_trials))

    with pytest.raises(ValueError):  # not at least 2d
        get_nondiag_weight(rs.randn(n_channels))
    with pytest.raises(ValueError):  # not square
        get_nondiag_weight(rs.randn(n_trials, n_channels, n_channels + 2))
