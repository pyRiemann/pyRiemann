import warnings
from functools import wraps

import numpy as np
from scipy.linalg import block_diag
from scipy.stats import chi2
from sklearn.covariance import oas, ledoit_wolf, fast_mcd

from .distance import distance_mahalanobis
from .test import is_square, is_real_type
from .utils import check_function


def _complex_estimator(func):
    """Decorator to extend a real-valued covariance estimator to complex data.

    Applied to a real-valued covariance estimator, this decorator allows to
    estimate complex covariance matrices from complex-valued multi-channel
    time-series. See Eq.(4) in [1]_.

    Parameters
    ----------
    func : callable
        Real-valued covariance estimator.

    Returns
    -------
    output : callable
        Complex-valued covariance estimator.

    Notes
    -----
    .. versionadded:: 0.6

    References
    ----------
    .. [1] `Enhanced Covariance Matrix Estimators in Adaptive Beamforming
        <https://doi.org/10.1109/ICASSP.2007.366399>`_
        R. Abrahamsson, Y. Selen and P. Stoica. 2007 IEEE International
        Conference on Acoustics, Speech and Signal Processing, Volume 2, 2007.
    """
    @wraps(func)
    def wrapper(X, **kwds):
        iscomplex = np.iscomplexobj(X)
        if iscomplex:
            n_channels, n_times = X.shape
            X = np.concatenate((X.real, X.imag), axis=0)
        cov = func(X, **kwds)
        if iscomplex:
            cov = cov[:n_channels, :n_channels] \
                + cov[n_channels:, n_channels:] \
                + 1j * (cov[n_channels:, :n_channels]
                        - cov[:n_channels, n_channels:])
        return cov
    return wrapper


@_complex_estimator
def _lwf(X, **kwds):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = ledoit_wolf(X.T, **kwds)
    return C


@_complex_estimator
def _mcd(X, **kwds):
    """Wrapper for sklearn mcd covariance estimator"""
    _, C, _, _ = fast_mcd(X.T, **kwds)
    return C


@_complex_estimator
def _oas(X, **kwds):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = oas(X.T, **kwds)
    return C


def _hub(X, **kwds):
    """Wrapper for Huber's M-estimator"""
    return covariance_mest(X, "hub", **kwds)


def _stu(X, **kwds):
    """Wrapper for Student-t's M-estimator"""
    return covariance_mest(X, "stu", **kwds)


def _tyl(X, **kwds):
    """Wrapper for Tyler's M-estimator"""
    return covariance_mest(X, "tyl", **kwds)


def covariance_mest(X, m_estimator, *, init=None, tol=10e-3, n_iter_max=50,
                    assume_centered=False, q=0.9, nu=5, norm="trace"):
    r"""Robust M-estimators.

    Robust M-estimator based covariance matrix [1]_, computed by fixed point
    algorithm.

    For an input time series :math:`X \in \mathbb{R}^{c \times t}`, composed of
    :math:`c` channels and :math:`t` time samples,

    .. math::
        C = \frac{1}{t} \sum_i \varphi(X[:,i]^H C^{-1} X[:,i]) X[:,i] X[:,i]^H

    where :math:`\varphi()` is a function allowing to weight the squared
    Mahalanobis distance depending on the M-estimator type: Huber, Student-t or
    Tyler.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series, real or complex-valued.
    m_estimator : {"hub", "stu", "tyl"}
        Type of M-estimator:

        - "hub" for Huber's M-estimator [2]_;
        - "stu" for Student-t's M-estimator [3]_;
        - "tyl" for Tyler's M-estimator [4]_.
    init : None | ndarray, shape (n_channels, n_channels), default=None
        A matrix used to initialize the algorithm.
        If None, the sample covariance matrix is used.
    tol : float, default=10e-3
        The tolerance to stop the fixed point estimation.
    n_iter_max : int, default=50
        The maximum number of iterations.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero. If False, data will be centered before computation.
    q : float, default=0.9
        Using Huber's M-estimator, q is the percentage in (0, 1] of inputs
        deemed uncorrupted, while (1-q) is the percentage of inputs treated as
        outliers w.r.t a Gaussian distribution.
        This estimator is a trade-off between Tyler's estimator (q=0) and the
        sample covariance matrix (q=1).
    nu : int, default=5
        Using Student-t's M-estimator, degree of freedom for t-distribution
        (strictly positive).
        This estimator is a trade-off between Tyler's estimator (nu->0) and the
        sample covariance matrix (nu->inf).
    norm : {"trace", "determinant"}, default="trace"
        Using Tyler's M-estimator, the type of normalization:

        * "trace": trace of covariance matrix is n_channels;
        * "determinant": determinant of covariance matrix is 1.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        Robust M-estimator based covariance matrix.

    Notes
    -----
    .. versionadded:: 0.4

    References
    ----------
    .. [1] `Complex Elliptically Symmetric Distributions: Survey, New Results
        and Applications
        <https://www.researchgate.net/profile/H-Vincent-Poor/publication/258658018_Complex_Elliptically_Symmetric_Distributions_Survey_New_Results_and_Applications/links/550480100cf24cee3a0150e2/Complex-Elliptically-Symmetric-Distributions-Survey-New-Results-and-Applications.pdf>`_
        E. Ollila, D.E. Tyler, V. Koivunen, H.V. Poor. IEEE Transactions on
        Signal Processing, 2012.
    .. [2] `Robust antenna array processing using M-estimators of
        pseudo-covariance
        <http://lib.tkk.fi/Diss/2010/isbn9789526030319/article5.pdf>`_
        E. Ollila, V. Koivunen. PIMRC, 2003.
    .. [3] `Influence functions for array covariance matrix estimators
        <https://ieeexplore.ieee.org/abstract/document/1289447/>`_
        E. Ollila, V. Koivunen. IEEE SSP, 2003.
    .. [4] `A distribution-free M-estimator of multivariate scatter
        <https://projecteuclid.org/journals/annals-of-statistics/volume-15/issue-1/A-Distribution-Free-M-Estimator-of-Multivariate-Scatter/10.1214/aos/1176350263.full>`_
        D.E. Tyler. The Annals of Statistics, 1987.
    """  # noqa
    n_channels, n_times = X.shape

    if m_estimator == "hub":
        if not 0 < q <= 1:
            raise ValueError(f"Value q must be included in (0, 1] (Got {q})")

        def weight_func(x):  # Example 1, Section V-C in [1]
            c2 = chi2.ppf(q, n_channels) / 2
            b = chi2.cdf(2 * c2, n_channels + 1) + c2 * (1 - q) / n_channels
            return np.minimum(1, c2 / x) / b
    elif m_estimator == "stu":
        if nu <= 0:
            raise ValueError(f"Value nu must be strictly positive (Got {nu})")

        def weight_func(x):  # Eq.(42) in [1]
            return (2 * n_channels + nu) / (nu + 2 * x)
    elif m_estimator == "tyl":
        def weight_func(x):  # Example 2, Section V-C in [1]
            return n_channels / x
    else:
        raise ValueError(f"Unsupported m_estimator: {m_estimator}")

    if not assume_centered:
        X -= np.mean(X, axis=1, keepdims=True)
    if init is None:
        cov = X @ X.conj().T / n_times
    else:
        cov = init

    for _ in range(n_iter_max):

        dist2 = distance_mahalanobis(X, cov, squared=True)
        Xw = np.sqrt(weight_func(dist2)) * X
        cov_new = Xw @ Xw.conj().T / n_times

        norm_delta = np.linalg.norm(cov_new - cov, ord="fro")
        norm_cov = np.linalg.norm(cov, ord="fro")
        cov = cov_new
        if (norm_delta / norm_cov) <= tol:
            break
    else:
        warnings.warn("Convergence not reached")

    if m_estimator == "tyl":
        cov = normalize(cov, norm)
        if norm == "trace":
            cov *= n_channels

    return cov


@_complex_estimator
def covariance_sch(X):
    r"""Schaefer-Strimmer shrunk covariance estimator.

    Shrinkage covariance estimator [1]_:

    .. math::
        C = (1 - \gamma) C_\text{scm} + \gamma T

    where :math:`T` is the diagonal target matrix:

    .. math::
        T[i,j] = \{ C_\text{scm}[i,i] \ \text{if} \ i=j,
                    0 \ \text{otherwise} \}

    Note that the optimal :math:`\gamma` is estimated by the authors' method.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series, real or complex-valued.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        Schaefer-Strimmer shrunk covariance matrix.

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `A shrinkage approach to large-scale covariance estimation and
        implications for functional genomics
        <http://doi.org/10.2202/1544-6115.1175>`_
        J. Schafer, and K. Strimmer. Statistical Applications in Genetics and
        Molecular Biology, Volume 4, Issue 1, 2005.
    """
    _, n_times = X.shape
    X_c = X - X.mean(axis=1, keepdims=True)
    C_scm = X_c @ X_c.T / n_times

    # Compute optimal gamma, the weigthing between SCM and shrinkage estimator
    R = (n_times / ((n_times - 1.) * np.outer(X.std(axis=1), X.std(axis=1))))
    R *= C_scm
    var_R = (X_c ** 2) @ (X_c ** 2).T - 2 * C_scm * (X_c @ X_c.T)
    var_R += n_times * C_scm ** 2
    Xvar = np.outer(X.var(axis=1), X.var(axis=1))
    var_R *= n_times / ((n_times - 1) ** 3 * Xvar)
    R -= np.diag(np.diag(R))
    var_R -= np.diag(np.diag(var_R))
    gamma = max(0, min(1, var_R.sum() / (R ** 2).sum()))

    sigma = (1. - gamma) * (n_times / (n_times - 1.)) * C_scm
    shrinkage = gamma * (n_times / (n_times - 1.)) * np.diag(np.diag(C_scm))
    return sigma + shrinkage


def covariance_scm(X, *, assume_centered=False):
    """Sample covariance estimator.

    Sample covariance estimator, re-implementing ``empirical_covariance`` of
    scikit-learn [1]_, but supporting real and complex-valued data.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series, real or complex-valued.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.

    Returns
    -------
    cov : ndarray, shape (n_channels, n_channels)
        Sample covariance matrix.

    Notes
    -----
    .. versionadded:: 0.6

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html
    """  # noqa
    _, n_times = X.shape

    if assume_centered:
        cov = X @ X.conj().T / n_times
    else:
        cov = np.cov(X, bias=1)

    return cov


###############################################################################


cov_est_functions = {
    "corr": np.corrcoef,
    "cov": np.cov,
    "hub": _hub,
    "lwf": _lwf,
    "mcd": _mcd,
    "oas": _oas,
    "sch": covariance_sch,
    "scm": covariance_scm,
    "stu": _stu,
    "tyl": _tyl,
}


def covariances(X, estimator="cov", **kwds):
    """Estimation of covariance matrices.

    Estimates covariance matrices from multi-channel time-series according to
    a covariance estimator. It supports real and complex-valued data.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_times)
        Multi-channel time-series, real or complex-valued.
    estimator : string | callable, default="cov"
        Covariance matrix estimator [est]_:

        * "corr" for correlation coefficient matrix [corr]_,
        * "cov" for NumPy based covariance matrix [cov]_,
        * "hub" for Huber's M-estimator based covariance matrix [mest]_,
        * "lwf" for Ledoit-Wolf shrunk covariance matrix [lwf]_,
        * "mcd" for minimum covariance determinant matrix [mcd]_,
        * "oas" for oracle approximating shrunk covariance matrix [oas]_,
        * "sch" for Schaefer-Strimmer shrunk covariance matrix [sch]_,
        * "scm" for sample covariance matrix [scm]_,
        * "stu" for Student-t's M-estimator based covariance matrix [mest]_,
        * "tyl" for Tyler's M-estimator based covariance matrix [mest]_,
        * or a callable function.

        For regularization, consider "lwf" or "oas".

        For robustness, consider "hub", "mcd", "stu" or "tyl".

        For "lwf", "mcd", "oas" and "sch" estimators,
        complex covariance matrices are estimated according to [comp]_.
    **kwds : dict
        Any further parameters are passed directly to the covariance estimator.

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Covariance matrices.

    References
    ----------
    .. [est] https://scikit-learn.org/stable/modules/covariance.html
    .. [corr] https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html
    .. [cov] https://numpy.org/doc/stable/reference/generated/numpy.cov.html
    .. [lwf] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html
    .. [mcd] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.MinCovDet.html
    .. [mest] :func:`pyriemann.utils.covariance.covariance_mest`
    .. [oas] https://scikit-learn.org/stable/modules/generated/oas-function.html
    .. [sch] :func:`pyriemann.utils.covariance.covariance_sch`
    .. [scm] :func:`pyriemann.utils.covariance.covariance_scm`
    .. [comp] `Enhanced Covariance Matrix Estimators in Adaptive Beamforming
        <https://doi.org/10.1109/ICASSP.2007.366399>`_
        R. Abrahamsson, Y. Selen and P. Stoica. 2007 IEEE International
        Conference on Acoustics, Speech and Signal Processing, Volume 2, 2007.
    """  # noqa
    est = check_function(estimator, cov_est_functions)
    n_matrices, n_channels, n_times = X.shape
    covmats = np.empty((n_matrices, n_channels, n_channels), dtype=X.dtype)
    for i in range(n_matrices):
        covmats[i] = est(X[i], **kwds)
    return covmats


def covariances_EP(X, P, estimator="cov", **kwds):
    """Special form covariance matrix, concatenating a prototype P.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_times)
        Multi-channel time-series.
    P : ndarray, shape (n_channels_proto, n_times)
        Multi-channel prototype.
    estimator : string | callable, default="cov"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the covariance estimator.

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels + n_channels_proto, \
            n_channels + n_channels_proto)
        Covariance matrices.
    """
    est = check_function(estimator, cov_est_functions)
    n_matrices, n_channels, n_times = X.shape
    n_channels_proto, n_times_p = P.shape
    if n_times_p != n_times:
        raise ValueError(
            f"X and P do not have the same n_times: {n_times} and {n_times_p}")
    covmats = np.empty((n_matrices, n_channels + n_channels_proto,
                        n_channels + n_channels_proto), dtype=X.dtype)
    for i in range(n_matrices):
        covmats[i] = est(np.concatenate((P, X[i]), axis=0), **kwds)
    return covmats


def covariances_X(X, estimator="cov", alpha=0.2, **kwds):
    """Special form covariance matrix, embedding input X.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_times)
        Multi-channel time-series.
    estimator : string | callable, default="cov"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    alpha : float, default=0.2
        Regularization parameter (strictly positive).
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the covariance estimator.

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels + n_times, n_channels + \
            n_times)
        Covariance matrices.

    References
    ----------
    .. [1] `A special form of SPD covariance matrix for interpretation and
        visualization of data manipulated with Riemannian geometry
        <https://hal.archives-ouvertes.fr/hal-01103344/>`_
        M. Congedo and A. Barachant, MaxEnt - 34th International Workshop on
        Bayesian Inference and Maximun Entropy Methods in Science and
        Engineering (MaxEnt'14), Sep 2014, Amboise, France. pp.495
    """
    if alpha <= 0:
        raise ValueError(
            f"Parameter alpha must be strictly positive (Got {alpha})")
    est = check_function(estimator, cov_est_functions)
    n_matrices, n_channels, n_times = X.shape

    Hchannels = np.eye(n_channels) \
        - np.outer(np.ones(n_channels), np.ones(n_channels)) / n_channels
    Htimes = np.eye(n_times) \
        - np.outer(np.ones(n_times), np.ones(n_times)) / n_times
    X = Hchannels @ X @ Htimes  # Eq(8), double centering

    covmats = np.empty(
        (n_matrices, n_channels + n_times, n_channels + n_times))
    for i in range(n_matrices):
        Y = np.concatenate((
            np.concatenate((X[i], alpha * np.eye(n_channels)), axis=1),  # top
            np.concatenate((alpha * np.eye(n_times), X[i].T), axis=1)  # bottom
        ), axis=0)  # Eq(9)
        covmats[i] = est(Y, **kwds)
    return covmats / (2 * alpha)  # Eq(10)


def block_covariances(X, blocks, estimator="cov", **kwds):
    """Compute block diagonal covariance.

    Calculates block diagonal matrices where each block is a covariance
    matrix of a subset of channels.
    Block sizes are passed as a list of integers and can vary. The sum
    of block sizes must equal the number of channels in X.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_times)
        Multi-channel time-series.
    blocks: list of int
        List of block sizes.
    estimator : string | callable, default="cov"
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    **kwds : optional keyword parameters
        Any further parameters are passed directly to the covariance estimator.

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Block diagonal covariance matrices.
    """
    est = check_function(estimator, cov_est_functions)
    n_matrices, n_channels, n_times = X.shape

    if np.sum(blocks) != n_channels:
        raise ValueError("Sum of individual block sizes "
                         "must match number of channels of X.")

    covmats = np.empty((n_matrices, n_channels, n_channels))
    for i in range(n_matrices):
        blockcov, idx_start = [], 0
        for j in blocks:
            blockcov.append(est(X[i, idx_start:idx_start+j, :], **kwds))
            idx_start += j
        covmats[i] = block_diag(*tuple(blockcov))

    return covmats


###############################################################################


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator="cov"):
    """Convert EEG signal to covariance using sliding window."""
    est = check_function(estimator, cov_est_functions)
    X = []
    if padding:
        padd = np.zeros((int(window / 2), sig.shape[1]))
        sig = np.concatenate((padd, sig, padd), axis=0)

    n_times, n_channels = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < n_times):
        X.append(est(sig[ix:ix + window, :].T))
        ix = ix + jump

    return np.array(X)


###############################################################################


def cross_spectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute the complex cross-spectral matrices of a real signal X.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series, real-valued.
    window : int, default=128
        Length of the FFT window used for spectral estimation.
    overlap : float, default=0.75
        Percentage of overlap between windows.
    fmin : float | None, default=None
        Minimal frequency to be returned.
    fmax : float | None, default=None
        Maximal frequency to be returned.
    fs : float | None, default=None
        Sampling frequency of the time-series.

    Returns
    -------
    S : ndarray, shape (n_channels, n_channels, n_freqs)
        Cross-spectral matrices, for each frequency bin.
    freqs : ndarray, shape (n_freqs,)
        Frequencies associated to cross-spectra.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Cross-spectrum
    """
    if not is_real_type(X):
        raise ValueError("Input must be real-valued.")
    window = int(window)
    if window < 1:
        raise ValueError("Value window must be a positive integer")
    if not 0 < overlap < 1:
        raise ValueError(
            f"Value overlap must be included in (0, 1) (Got {overlap})"
        )

    n_channels, n_times = X.shape
    n_freqs = int(window / 2) + 1  # X real signal => compute half-spectrum
    step = int((1.0 - overlap) * window)
    n_windows = int((n_times - window) / step + 1)
    win = np.hanning(window)

    # FFT calculation on all windows
    shape = (n_channels, n_windows, window)
    strides = X.strides[:-1]+(step*X.strides[-1], X.strides[-1])
    Xs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
    fdata = np.fft.rfft(Xs * win, n=window).transpose(1, 0, 2)

    # adjust frequency range to specified range
    if fs is not None:
        if fmin is None:
            fmin = 0
        if fmax is None:
            fmax = fs / 2
        if fmax <= fmin:
            raise ValueError("Parameter fmax must be superior to fmin")
        if 2.0 * fmax > fs:  # check Nyquist-Shannon
            raise ValueError("Parameter fmax must be inferior to fs/2")
        f = np.arange(0, n_freqs, dtype=int) * float(fs / window)
        fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[:, :, fix]
        freqs = f[fix]
    else:
        if fmin is not None:
            warnings.warn("Parameter fmin not used because fs is None")
        if fmax is not None:
            warnings.warn("Parameter fmax not used because fs is None")
        freqs = None

    n_freqs = fdata.shape[2]
    S = np.zeros((n_channels, n_channels, n_freqs), dtype=complex)
    for i in range(n_freqs):
        S[:, :, i] = fdata[:, :, i].conj().T @ fdata[:, :, i]
    S /= n_windows * np.linalg.norm(win)**2

    # normalization to respect Parseval's theorem with the half-spectrum
    # excepted DC bin (always), and Nyquist bin (when window is even)
    if window % 2:
        S[..., 1:] *= 2
    else:
        S[..., 1:-1] *= 2

    return S, freqs


def cospectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute co-spectral matrices, the real part of cross-spectra.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series, real-valued.
    window : int, default=128
        Length of the FFT window used for spectral estimation.
    overlap : float, default=0.75
        Percentage of overlap between windows.
    fmin : float | None, default=None
        Minimal frequency to be returned.
    fmax : float | None, default=None
        Maximal frequency to be returned.
    fs : float | None, default=None
        Sampling frequency of the time-series.

    Returns
    -------
    S : ndarray, shape (n_channels, n_channels, n_freqs)
        Co-spectral matrices, for each frequency bin.
    freqs : ndarray, shape (n_freqs,)
        Frequencies associated to cospectra.
    """
    S, freqs = cross_spectrum(
        X=X,
        window=window,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        fs=fs,
    )

    return S.real, freqs


def coherence(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None,
              coh="ordinary"):
    """Compute squared coherence.

    Parameters
    ----------
    X : ndarray, shape (n_channels, n_times)
        Multi-channel time-series, real-valued.
    window : int, default=128
        Length of the FFT window used for spectral estimation.
    overlap : float, default=0.75
        Percentage of overlap between windows.
    fmin : float | None, default=None
        Minimal frequency to be returned.
    fmax : float | None, default=None
        Maximal frequency to be returned.
    fs : float | None, default=None
        Sampling frequency of the time-series.
    coh : {"ordinary", "instantaneous", "lagged", "imaginary"}, \
            default="ordinary"
        Coherence type, see :class:`pyriemann.estimation.Coherences`.

    Returns
    -------
    C : ndarray, shape (n_channels, n_channels, n_freqs)
        Squared coherence matrices, for each frequency bin.
    freqs : ndarray, shape (n_freqs,)
        Frequencies associated to coherence.
    """
    S, freqs = cross_spectrum(
        X,
        window=window,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        fs=fs,
    )
    S2 = np.abs(S)**2  # squared cross-spectral modulus

    C = np.zeros_like(S2)
    f_inds = np.arange(0, C.shape[-1], dtype=int)

    # lagged coh not defined for DC and Nyquist bins, because S is real
    if coh == "lagged":
        if freqs is None:
            f_inds = np.arange(1, C.shape[-1] - 1, dtype=int)
            warnings.warn("DC and Nyquist bins are not defined for lagged-"
                          "coherence: filled with zeros")
        else:
            f_inds_ = f_inds[(freqs > 0) & (freqs < fs / 2)]
            if not np.array_equal(f_inds_, f_inds):
                warnings.warn("DC and Nyquist bins are not defined for lagged-"
                              "coherence: filled with zeros")
            f_inds = f_inds_

    for f in f_inds:
        psd = np.sqrt(np.diag(S2[..., f]))
        psd_prod = np.outer(psd, psd)
        if coh == "ordinary":
            C[..., f] = S2[..., f] / psd_prod
        elif coh == "instantaneous":
            C[..., f] = (S[..., f].real)**2 / psd_prod
        elif coh == "lagged":
            np.fill_diagonal(S[..., f].real, 0.)  # prevent div by zero on diag
            denom = psd_prod - (S[..., f].real)**2
            denom[abs(denom) < 1e-10] = 1e-10
            C[..., f] = (S[..., f].imag)**2 / denom
        elif coh == "imaginary":
            C[..., f] = (S[..., f].imag)**2 / psd_prod
        else:
            raise ValueError(f"{coh} is not a supported coherence")

    return C, freqs


###############################################################################


def normalize(X, norm):
    """Normalize a set of square matrices, using corr, trace or determinant.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Set of square matrices, at least 2D ndarray.
        Matrices must be invertible for determinant-normalization.

    norm : {"corr", "trace", "determinant"}
        Type of normalization:

        * "corr": normalized matrices are correlation matrices, with values in
          [-1, 1] and diagonal values equal to 1;
        * "trace": trace of normalized matrices is 1;
        * "determinant": determinant of normalized matrices is +/- 1.

    Returns
    -------
    Xn : ndarray, shape (..., n, n)
        Set of normalized matrices, same dimensions as X.
    """
    if not is_square(X):
        raise ValueError("Matrices must be square")

    if norm == "corr":
        stddev = np.sqrt(np.abs(np.diagonal(X, axis1=-2, axis2=-1)))
        denom = np.expand_dims(stddev, axis=-2) * stddev[..., np.newaxis]
    elif norm == "trace":
        denom = np.trace(X, axis1=-2, axis2=-1)
    elif norm == "determinant":
        denom = np.abs(np.linalg.det(X)) ** (1 / X.shape[-1])
    else:
        raise ValueError(f"{norm} is not a supported normalization")

    denom = np.expand_dims(denom, axis=tuple(range(denom.ndim, X.ndim)))
    Xn = X / denom

    if norm == "corr":
        np.clip(Xn, -1, 1, out=Xn)

    return Xn


def get_nondiag_weight(X):
    """Compute non-diagonality weights of a set of square matrices.

    Compute non-diagonality weights of a set of square matrices, following
    Eq(B.1) in [1]_.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        Set of square matrices, at least 2D ndarray.

    Returns
    -------
    weights : ndarray, shape (...,)
        Non-diagonality weights for matrices.

    References
    ----------
    .. [1] `On the blind source separation of human electroencephalogram by
        approximate joint diagonalization of second order statistics
        <https://hal.archives-ouvertes.fr/hal-00343628>`_
        M. Congedo, C. Gouy-Pailler, C. Jutten. Clinical Neurophysiology,
        Elsevier, 2008, 119 (12), pp.2677-2686.
    """
    if not is_square(X):
        raise ValueError("Matrices must be square")

    X2 = X**2
    # sum of squared diagonal elements
    denom = np.trace(X2, axis1=-2, axis2=-1)
    # sum of squared off-diagonal elements
    num = np.sum(X2, axis=(-2, -1)) - denom
    weights = (1.0 / (X.shape[-1] - 1)) * (num / denom)
    return weights
