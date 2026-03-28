from functools import wraps
import warnings

import numpy as np
from scipy.stats import chi2
from sklearn import config_context
from sklearn.covariance import oas, ledoit_wolf, fast_mcd

from ._backend import (
    get_namespace, is_numpy_namespace,
    create_diagonal, diag_indices, xpd,
)
from .base import ctranspose, _vectorize_nd
from .distance import distance_mahalanobis
from .test import is_square, is_real_type
from .utils import check_function, check_init, check_weights

try:  # pragma: no cover - torch is optional
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None


def _to_numpy(X):
    return X if isinstance(X, np.ndarray) else X.detach().cpu().numpy()


def _from_numpy(X, *, like):
    xp = get_namespace(like)
    return xp.asarray(X, dtype=like.dtype, device=xpd(like))


def _cov(X, **kwds):
    xp = get_namespace(X)
    if is_numpy_namespace(xp):
        C = np.cov(X, **kwds)
        return np.atleast_2d(C)
    if kwds:
        # map np.cov kwargs to torch.cov kwargs
        correction = kwds.pop("ddof", 1)
        if kwds.pop("bias", False):
            correction = 0
        return xp.cov(X, correction=correction, **kwds)
    X_c = X - xp.mean(X, axis=1)[:, xp.newaxis]
    return X_c @ ctranspose(X_c) / (X.shape[1] - 1)


def _corr(X, **kwds):
    xp = get_namespace(X)
    if is_numpy_namespace(xp):
        return np.corrcoef(X, **kwds) if kwds else normalize(_cov(X), "corr")
    if kwds:
        return xp.corrcoef(X)
    return normalize(_cov(X), "corr")


def _make_complex(real_part, imag_part, *, like):
    if isinstance(like, np.ndarray):
        return real_part + 1j * imag_part
    return torch.complex(real_part, imag_part)


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
        xp = get_namespace(X)
        iscomplex = not is_real_type(X)
        if iscomplex:
            n_channels = X.shape[-2]
            X = xp.concat(
                (xp.real(X), xp.imag(X)),
                axis=-2,
            )
        cov = func(X, **kwds)
        if iscomplex:
            cov = _make_complex(
                cov[..., :n_channels, :n_channels]
                + cov[..., n_channels:, n_channels:],
                cov[..., n_channels:, :n_channels]
                - cov[..., :n_channels, n_channels:],
                like=X,
            )
        return cov
    return wrapper


def _sklearn_array_api(func, X, **kwds):
    """Call sklearn estimator with array API dispatch if available."""
    xp = get_namespace(X)
    if is_numpy_namespace(xp):
        return func(X.mT, **kwds)
    try:
        with config_context(array_api_dispatch=True):
            return func(X.mT, **kwds)
    except RuntimeError:
        # SCIPY_ARRAY_API not set — fall back to numpy conversion
        result = func(_to_numpy(X).mT, **kwds)
        if isinstance(result, tuple):
            return tuple(
                _from_numpy(r, like=X) if hasattr(r, 'shape') else r
                for r in result
            )
        return _from_numpy(result, like=X)


@_complex_estimator
def _lwf(X, **kwds):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    C, _ = _sklearn_array_api(ledoit_wolf, X, **kwds)
    return C


@_complex_estimator
def _mcd(X, **kwds):
    """Wrapper for sklearn mcd covariance estimator"""
    # fast_mcd does not support array API yet
    _, C, _, _ = fast_mcd(_to_numpy(X).mT, **kwds)
    return C if isinstance(X, np.ndarray) else _from_numpy(C, like=X)


@_complex_estimator
def _oas(X, **kwds):
    """Wrapper for sklearn oas covariance estimator"""
    C, _ = _sklearn_array_api(oas, X, **kwds)
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
    X : ndarray, shape (..., n_channels, n_times)
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
    cov : ndarray, shape (..., n_channels, n_channels)
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
    xp = get_namespace(X, init)
    n_channels, n_times = X.shape[-2], X.shape[-1]

    if m_estimator == "hub":
        if not 0 < q <= 1:
            raise ValueError(f"Value q must be included in (0, 1] (Got {q})")

        def weight_func(x):  # Example 1, Section V-C in [1]
            c2 = chi2.ppf(q, n_channels) / 2
            b = chi2.cdf(2 * c2, n_channels + 1) + c2 * (1 - q) / n_channels
            return xp.minimum(
                xp.ones(x.shape, dtype=x.real.dtype, device=xpd(x)),
                c2 / x,
            ) / b
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
        X = X - xp.mean(X, axis=-1, keepdims=True)
    if init is None:
        cov = X @ ctranspose(X) / n_times
    else:
        cov = check_init(init, n_channels, like=X)

    for _ in range(n_iter_max):
        dist2 = distance_mahalanobis(X, cov, squared=True)
        Xw = xp.sqrt(weight_func(dist2))[np.newaxis, :] * X
        cov_new = Xw @ ctranspose(Xw) / n_times

        norm_delta = float(xp.linalg.matrix_norm(cov_new - cov))
        norm_cov = float(xp.linalg.matrix_norm(cov))
        cov = cov_new
        if (norm_delta / norm_cov) <= tol:
            break
    else:
        warnings.warn("Convergence not reached", stacklevel=2)

    if m_estimator == "tyl":
        cov = normalize(cov, norm)
        if norm == "trace":
            cov *= n_channels

    return cov


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
    X : ndarray, shape (..., n_channels, n_times)
        Multi-channel time-series, real or complex-valued.

    Returns
    -------
    cov : ndarray, shape (..., n_channels, n_channels)
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
    xp = get_namespace(X)
    if not is_real_type(X):
        n_channels = X.shape[-2]
        X_ri = xp.concat(
            (xp.real(X), xp.imag(X)),
            axis=-2,
        )
        cov = covariance_sch(X_ri)
        return (
            cov[..., :n_channels, :n_channels]
            + cov[..., n_channels:, n_channels:]
            + 1j * (
                cov[..., n_channels:, :n_channels]
                - cov[..., :n_channels, n_channels:]
            )
        )

    n_times = X.shape[-1]
    X_c = X - xp.mean(X, axis=-1, keepdims=True)
    C_scm = X_c @ ctranspose(X_c) / n_times

    # Compute optimal gamma, the weighting between SCM and shrinkage estimator
    std = (
        X.std(axis=-1)
        if isinstance(X, np.ndarray)
        else torch.std(X, dim=-1, correction=0)
    )
    R = n_times / ((n_times - 1.) * xp.linalg.outer(std, std))
    R *= C_scm
    var_R = (X_c ** 2) @ ctranspose(X_c ** 2)
    var_R -= 2 * C_scm * (X_c @ ctranspose(X_c))
    var_R += n_times * C_scm ** 2
    var = (
        X.var(axis=-1)
        if isinstance(X, np.ndarray)
        else torch.var(X, dim=-1, correction=0)
    )
    Xvar = xp.linalg.outer(var, var)
    var_R *= n_times / ((n_times - 1) ** 3 * Xvar)
    diag0, diag1 = diag_indices(R.shape[-1], xp=xp, like=R)
    R[diag0, diag1] = 0
    var_R[diag0, diag1] = 0
    denom = float(xp.sum(R ** 2))
    gamma = 0 if denom == 0 else max(
        0,
        min(
            1,
            float(xp.sum(var_R)) / denom,
        ),
    )

    sigma = (1. - gamma) * (n_times / (n_times - 1.)) * C_scm
    shrinkage = (
        gamma
        * (n_times / (n_times - 1.))
        * create_diagonal(xp.linalg.diagonal(C_scm))
    )
    return sigma + shrinkage


def covariance_scm(X, *, assume_centered=False, weights=None):
    r"""Sample covariance estimator.

    Sample covariance estimator, re-implementing ``empirical_covariance`` of
    scikit-learn [1]_, but supporting:

    - real and complex-valued data,
    - broadcasting,
    - weights for time samples.

    .. math::
        \mathbf{C}_\text{scm} = \mathbf{X} \text{diag}(w) \mathbf{X}^H

    with :math:`w` being the weights which sum to 1.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_times)
        Multi-channel time-series, real or complex-valued.
    assume_centered : bool, default=False
        If True, data will not be centered before computation.
        Useful when working with data whose mean is almost, but not exactly
        zero.
        If False, data will be centered before computation.
    weights : None | ndarray, shape (n_times,), default=None
        Weights for each time sample. If None, it uses equal weights.

        .. versionadded:: 0.11

    Returns
    -------
    cov : ndarray, shape (..., n_channels, n_channels)
        Sample covariance matrix.

    Notes
    -----
    .. versionadded:: 0.6

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn.covariance.empirical_covariance.html
    """  # noqa
    xp = get_namespace(X)
    weights = check_weights(weights, X.shape[-1], like=X)

    if not assume_centered:
        X = X - xp.sum(X * weights, axis=-1, keepdims=True)
    return (X * weights) @ ctranspose(X)


###############################################################################


cov_est_functions = {
    "corr": _corr,
    "cov": _cov,
    "hub": _hub,
    "lwf": _lwf,
    "mcd": _mcd,
    "oas": _oas,
    "sch": covariance_sch,
    "scm": covariance_scm,
    "stu": _stu,
    "tyl": _tyl,
}


def _check_cov_estimator(estimator):
    est = check_function(estimator, cov_est_functions)
    est = _vectorize_nd(batch_native=False)(est)
    return est


def covariances(X, estimator="cov", **kwds):
    """Estimation of covariance matrices.

    Estimates covariance matrices from multi-channel time-series according to
    a covariance estimator. It supports real and complex-valued data.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_times)
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
    covmats : ndarray, shape (..., n_channels, n_channels)
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
    est = _check_cov_estimator(estimator)
    return est(X, **kwds)


def covariances_EP(X, P, estimator="cov", **kwds):
    """Special form covariance matrix, concatenating a prototype P.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_times)
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
    covmats : ndarray, shape (..., n_channels + n_channels_proto, \
            n_channels + n_channels_proto)
        Covariance matrices.
    """
    est = _check_cov_estimator(estimator)
    xp = get_namespace(X, P)
    original_shape = X.shape
    n_times = original_shape[-1]
    n_channels_proto, n_times_p = P.shape
    if n_times_p != n_times:
        raise ValueError(
            f"X and P do not have the same n_times: {n_times} and {n_times_p}")
    P_broadcast = xp.broadcast_to(
        P, (*original_shape[:-2], n_channels_proto, n_times)
    )
    PX = xp.concat((P_broadcast, X), axis=-2)
    return est(PX, **kwds)


def covariances_X(X, estimator="cov", alpha=0.2, **kwds):
    """Special form covariance matrix, embedding input X.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_times)
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
    covmats : ndarray, shape (..., n_channels + n_times, n_channels + \
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
    est = _check_cov_estimator(estimator)
    xp = get_namespace(X)
    original_shape = X.shape
    n_channels, n_times = original_shape[-2], original_shape[-1]

    dt, dev = X.real.dtype, xpd(X)
    ones_ch = xp.ones(n_channels, dtype=dt, device=dev)
    Hchannels = (
        xp.eye(n_channels, dtype=dt, device=dev)
        - xp.linalg.outer(ones_ch, ones_ch) / n_channels
    )
    ones_t = xp.ones(n_times, dtype=dt, device=dev)
    Htimes = (
        xp.eye(n_times, dtype=dt, device=dev)
        - xp.linalg.outer(ones_t, ones_t) / n_times
    )
    X = Hchannels @ X @ Htimes  # Eq(8), double centering

    batch_shape = original_shape[:-2]
    I_ch = xp.broadcast_to(
        alpha * xp.eye(n_channels, dtype=X.dtype, device=xpd(X)),
        (*batch_shape, n_channels, n_channels),
    )
    I_t = xp.broadcast_to(
        alpha * xp.eye(n_times, dtype=X.dtype, device=xpd(X)),
        (*batch_shape, n_times, n_times),
    )
    Xt = X.mT
    top = xp.concat([X, I_ch], axis=-1)
    bot = xp.concat([I_t, Xt], axis=-1)
    Y = xp.concat([top, bot], axis=-2)  # Eq(9)
    return est(Y, **kwds) / (2 * alpha)  # Eq(10)


def block_covariances(X, blocks, estimator="cov", **kwds):
    """Compute block diagonal covariance.

    Calculates block diagonal matrices where each block is a covariance
    matrix of a subset of channels.
    Block sizes are passed as a list of integers and can vary. The sum
    of block sizes must equal the number of channels in X.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_times)
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
    covmats : ndarray, shape (..., n_channels, n_channels)
        Block diagonal covariance matrices.
    """
    est = _check_cov_estimator(estimator)
    xp = get_namespace(X)
    original_shape = X.shape
    n_channels = original_shape[-2]

    if sum(blocks) != n_channels:
        raise ValueError("Sum of individual block sizes "
                         "must match number of channels of X.")

    covmats = xp.zeros(
        (*original_shape[:-2], n_channels, n_channels),
        dtype=X.dtype,
        device=xpd(X),
    )
    idx_start = 0
    for j in blocks:
        block_cov = est(X[..., idx_start:idx_start+j, :], **kwds)
        if block_cov.ndim < 2:
            block_cov = xp.reshape(block_cov, (*block_cov.shape, 1, 1))
        covmats[..., idx_start:idx_start+j, idx_start:idx_start+j] = \
            block_cov
        idx_start += j
    return covmats


###############################################################################


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator="cov"):
    """Convert EEG signal to covariance using sliding window."""
    est = check_function(estimator, cov_est_functions)
    xp = get_namespace(sig)
    X = []
    if padding:
        padd = xp.zeros(
            (int(window / 2), sig.shape[1]),
            dtype=sig.dtype, device=xpd(sig),
        )
        sig = xp.concat((padd, sig, padd), axis=0)

    n_times, n_channels = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < n_times):
        X.append(est(sig[ix:ix + window, :].mT))
        ix = ix + jump

    return xp.stack(X, axis=0)


###############################################################################


def cross_spectrum(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None):
    """Compute the complex cross-spectral matrices of a real signal X.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_times)
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
    S : ndarray, shape (..., n_channels, n_channels, n_freqs)
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

    n_times = X.shape[-1]
    n_freqs = int(window / 2) + 1  # X real signal => compute half-spectrum
    step = int((1.0 - overlap) * window)
    n_windows = int((n_times - window) / step + 1)
    win = np.hanning(window)

    # Sliding window view handles any batch dims
    # (..., n_channels, n_times) -> (..., n_channels, n_windows, window)
    Xs = np.lib.stride_tricks.sliding_window_view(
        X, window, axis=-1
    )[..., ::step, :]
    # FFT: (..., n_channels, n_windows, n_freqs)
    fdata = np.fft.rfft(Xs * win, n=window)

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
        if isinstance(X, np.ndarray):
            f = np.arange(0, n_freqs, dtype=int) * float(fs / window)
        else:
            f = torch.arange(
                0,
                n_freqs,
                device=X.device,
                dtype=X.real.dtype,
            ) * float(fs / window)
        fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[..., fix]
        freqs = f[fix]
    else:
        if fmin is not None:
            warnings.warn(
                "Parameter fmin not used because fs is None",
                stacklevel=2,
            )
        if fmax is not None:
            warnings.warn(
                "Parameter fmax not used because fs is None",
                stacklevel=2,
            )
        freqs = None

    # Cross-spectral matrix via einsum over windows
    # (..., n_channels, n_channels, n_freqs)
    S = np.einsum('...cwf,...dwf->...cdf', fdata.conj(), fdata)
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
    X : ndarray, shape (..., n_channels, n_times)
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
    S : ndarray, shape (..., n_channels, n_channels, n_freqs)
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

    return get_namespace(S).real(S), freqs


def coherence(X, window=128, overlap=0.75, fmin=None, fmax=None, fs=None,
              coh="ordinary"):
    """Compute squared coherence.

    Parameters
    ----------
    X : ndarray, shape (..., n_channels, n_times)
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
        Coherence type:

        * "ordinary" for the ordinary coherence, defined in Eq.(22) of [1]_;
          this normalization of cross-spectral matrices captures both in-phase
          and out-of-phase correlations. However it is inflated by the
          artificial in-phase (zero-lag) correlation engendered by volume
          conduction.
        * "instantaneous" for the instantaneous coherence, Eq.(26) of [1]_,
          capturing only in-phase correlation.
        * "lagged" for the lagged-coherence, Eq.(28) of [1]_, capturing only
          out-of-phase correlation (not defined for DC and Nyquist bins).
        * "imaginary" for the imaginary coherence [2]_, Eq.(0.16) of [3]_,
          capturing out-of-phase correlation but still affected by in-phase
          correlation.

    Returns
    -------
    C : ndarray, shape (..., n_channels, n_channels, n_freqs)
        Squared coherence matrices, for each frequency bin.
    freqs : ndarray, shape (n_freqs,)
        Frequencies associated to coherence.

    References
    ----------
    .. [1] `Instantaneous and lagged measurements of linear
        and nonlinear dependence between groups of multivariate time series:
        frequency decomposition
        <https://arxiv.org/ftp/arxiv/papers/0711/0711.1455.pdf>`_
        R. Pascual-Marqui. Technical report, 2007.
    .. [2] `Identifying true brain interaction from EEG data using the
        imaginary part of coherency
        <https://doi.org/10.1016/j.clinph.2004.04.029>`_
        G. Nolte, O. Bai, L. Wheaton, Z. Mari, S. Vorbach, M. Hallett.
        Clinical Neurophysioly, Volume 115, Issue 10, October 2004,
        Pages 2292-2307
    .. [3] `Non-Parametric Synchronization Measures used in EEG
        and MEG
        <https://hal.archives-ouvertes.fr/hal-01868538v2>`_
        M. Congedo. Technical Report, 2018.
    """
    S, freqs = cross_spectrum(
        X,
        window=window,
        overlap=overlap,
        fmin=fmin,
        fmax=fmax,
        fs=fs,
    )
    # S: (..., n_channels, n_channels, n_freqs)
    S2 = np.abs(S) ** 2  # squared cross-spectral modulus

    n_channels = S.shape[-3]
    C = np.zeros_like(S2)
    f_inds = np.arange(0, C.shape[-1], dtype=int)

    # lagged coh not defined for DC and Nyquist bins, because S is real
    if coh == "lagged":
        if freqs is None:
            f_inds = np.arange(1, C.shape[-1] - 1, dtype=int)
            warnings.warn(
                "DC and Nyquist bins are not defined for lagged-"
                "coherence: filled with zeros",
                stacklevel=2,
            )
        else:
            f_inds_ = f_inds[(freqs > 0) & (freqs < fs / 2)]
            if not np.array_equal(f_inds_, f_inds):
                warnings.warn(
                    "DC and Nyquist bins are not defined for lagged-"
                    "coherence: filled with zeros",
                    stacklevel=2,
                )
            f_inds = f_inds_

    # Vectorized PSD: diagonal of S2 across channels
    diag_idx = np.arange(n_channels)
    psd = np.sqrt(S2[..., diag_idx, diag_idx, :])  # (..., n_ch, n_freqs)
    psd_prod = (
        psd[..., :, np.newaxis, :]
        * psd[..., np.newaxis, :, :]
    )  # (..., n_ch, n_ch, n_freqs)

    if coh == "ordinary":
        C[..., f_inds] = S2[..., f_inds] / psd_prod[..., f_inds]
    elif coh == "instantaneous":
        C[..., f_inds] = (
            S[..., f_inds].real ** 2 / psd_prod[..., f_inds]
        )
    elif coh == "lagged":
        S_real = S.real.copy()
        S_real[..., diag_idx, diag_idx, :] = 0.0  # zero diagonal
        denom = psd_prod[..., f_inds] - S_real[..., f_inds] ** 2
        denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
        C[..., f_inds] = S[..., f_inds].imag ** 2 / denom
    elif coh == "imaginary":
        C[..., f_inds] = (
            S[..., f_inds].imag ** 2 / psd_prod[..., f_inds]
        )
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
    xp = get_namespace(X)
    if not is_square(X):
        raise ValueError("Matrices must be square")

    if norm == "corr":
        stddev = xp.sqrt(xp.abs(xp.linalg.diagonal(X)))
        denom = stddev[..., :, np.newaxis] * stddev[..., np.newaxis, :]
    elif norm == "trace":
        denom = xp.sum(xp.linalg.diagonal(X), axis=-1)
    elif norm == "determinant":
        _, logabsdet = xp.linalg.slogdet(X)
        denom = xp.exp(logabsdet / X.shape[-1])
    else:
        raise ValueError(f"{norm} is not a supported normalization")

    while denom.ndim < X.ndim:
        denom = denom[..., np.newaxis]
    Xn = X / denom

    if norm == "corr":
        if not is_real_type(Xn):
            return Xn
        if isinstance(Xn, np.ndarray):
            np.clip(Xn, -1, 1, out=Xn)
        else:
            Xn = torch.clamp(Xn, -1, 1)

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
    xp = get_namespace(X)
    if not is_square(X):
        raise ValueError("Matrices must be square")

    X2 = X**2
    # sum of squared diagonal elements
    denom = xp.sum(xp.linalg.diagonal(X2), axis=-1)
    # sum of squared off-diagonal elements
    num = xp.sum(X2, axis=(-2, -1)) - denom
    weights = (1.0 / (X.shape[-1] - 1)) * (num / denom)
    return weights
