import warnings
from functools import wraps

import numpy as np
from scipy.stats import chi2
from sklearn.covariance import oas, ledoit_wolf, fast_mcd
from sklearn.utils import check_random_state

from ._backend import resolve_backend
from .base import ctranspose
from .distance import distance_mahalanobis
from .test import is_square, is_real_type
from .utils import check_function, check_init

try:  # pragma: no cover - torch is optional
    import torch
except ImportError:  # pragma: no cover - torch is optional
    torch = None


def _to_numpy(X):
    return X if isinstance(X, np.ndarray) else X.detach().cpu().numpy()


def _from_numpy(X, *, like):
    backend = resolve_backend(like)
    return backend.asarray(X, like=like, dtype=like.dtype)


def _apply_numpy_estimator(func, X, **kwds):
    cov = func(_to_numpy(X), **kwds)
    return cov if isinstance(X, np.ndarray) else _from_numpy(cov, like=X)


def _cov(X, **kwds):
    if kwds:
        return _apply_numpy_estimator(np.cov, X, **kwds)
    backend = resolve_backend(X)
    X_c = X - backend.mean(X, axis=1)[:, np.newaxis]
    return X_c @ ctranspose(X_c, backend=backend) / (X.shape[1] - 1)


def _corr(X, **kwds):
    if kwds or not is_real_type(X):
        return _apply_numpy_estimator(np.corrcoef, X, **kwds)
    return normalize(_cov(X), "corr")


def _make_complex(real_part, imag_part, *, like):
    if isinstance(like, np.ndarray):
        return real_part + 1j * imag_part
    return torch.complex(real_part, imag_part)


def _as_samples_by_features(X, *, assume_centered=False, backend=None):
    backend = resolve_backend(X, backend=backend)
    X = backend.swapaxes(X, 0, 1)
    if not assume_centered:
        X = X - backend.mean(X, axis=0)
    return X, backend


def _empirical_covariance_features(X, *, assume_centered=False, backend=None):
    backend = resolve_backend(X, backend=backend)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.shape[0] == 1:
        warnings.warn(
            "Only one sample available. You may want to reshape your data "
            "array",
            stacklevel=2,
        )

    if assume_centered:
        covariance = ctranspose(X, backend=backend) @ X / X.shape[0]
    else:
        X = X - backend.mean(X, axis=0)
        covariance = ctranspose(X, backend=backend) @ X / X.shape[0]

    if covariance.ndim == 0:
        covariance = covariance.reshape(1, 1)
    return covariance


def _shrink_covariance(emp_cov, shrinkage, *, backend=None):
    backend = resolve_backend(emp_cov, backend=backend)
    n_features = emp_cov.shape[-1]
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    mu = backend.as_float(backend.sum(backend.diagonal(emp_cov)) / n_features)
    diag0, diag1 = backend.diag_indices(n_features, like=emp_cov)
    shrunk_cov[diag0, diag1] += shrinkage * mu
    return shrunk_cov


def _one_feature_covariance(X_samples, *, backend):
    cov = backend.mean(X_samples ** 2, axis=0).reshape(1, 1)
    return cov, 0.0


def _ledoit_wolf_torch(X, *, assume_centered=False, block_size=1000):
    del block_size  # kept for API compatibility
    X_samples, backend = _as_samples_by_features(
        X,
        assume_centered=assume_centered,
    )

    if X_samples.shape[1] == 1:
        return _one_feature_covariance(X_samples, backend=backend)

    n_samples, n_features = X_samples.shape
    X2 = X_samples ** 2
    emp_cov_trace = backend.sum(X2, axis=0) / n_samples
    mu = backend.as_float(backend.sum(emp_cov_trace) / n_features)
    beta_ = backend.as_float(
        backend.sum(ctranspose(X2, backend=backend) @ X2)
    )
    gram = ctranspose(X_samples, backend=backend) @ X_samples
    delta_ = backend.as_float(backend.sum(gram ** 2)) / (n_samples ** 2)
    beta = (beta_ / n_samples - delta_) / (n_features * n_samples)
    delta = (
        delta_
        - 2.0 * mu * backend.as_float(backend.sum(emp_cov_trace))
        + n_features * mu ** 2
    ) / n_features
    beta = min(beta, delta)
    shrinkage = 0.0 if beta == 0 else beta / delta
    emp_cov = covariance_scm(X, assume_centered=assume_centered)
    return _shrink_covariance(emp_cov, shrinkage, backend=backend), shrinkage


def _oas_torch(X, *, assume_centered=False):
    X_samples, backend = _as_samples_by_features(
        X,
        assume_centered=assume_centered,
    )

    if X_samples.shape[1] == 1:
        return _one_feature_covariance(X_samples, backend=backend)

    n_samples, n_features = X_samples.shape
    emp_cov = covariance_scm(X, assume_centered=assume_centered)
    alpha = backend.as_float(backend.sum(emp_cov ** 2)) / (n_features ** 2)
    mu = backend.as_float(backend.sum(backend.diagonal(emp_cov)) / n_features)
    mu_squared = mu ** 2
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage = 1.0 if den == 0 else min(num / den, 1.0)
    return _shrink_covariance(emp_cov, shrinkage, backend=backend), shrinkage


def _fast_logdet_backend(A, *, backend=None):
    backend = resolve_backend(A, backend=backend)
    sign, ld = backend.slogdet(A)
    if backend.as_float(sign) <= 0 or not np.isfinite(backend.as_float(ld)):
        return -np.inf
    return backend.as_float(ld)


def _mahalanobis_dist_sq_features(X, location, covariance):
    precision = torch.linalg.pinv(covariance, hermitian=True)
    X_centered = X - location
    dist = torch.sum((X_centered @ precision) * X_centered, dim=1)
    return dist, precision


def _support_from_dist_torch(dist, n_support):
    support = torch.zeros(dist.shape[0], dtype=torch.bool, device=dist.device)
    support[torch.argsort(dist)[:n_support]] = True
    return support


def _mcd_1d_torch(X, n_support):
    X = X.reshape(-1)
    n_samples = X.shape[0]

    if n_support < n_samples:
        X_sorted, _ = torch.sort(X)
        diff = X_sorted[n_support:] - X_sorted[:(n_samples - n_support)]
        halves_start = torch.nonzero(
            diff == torch.min(diff),
            as_tuple=False,
        ).flatten()
        location = 0.5 * (
            X_sorted[n_support + halves_start] + X_sorted[halves_start]
        ).mean()
        support = _support_from_dist_torch(torch.abs(X - location), n_support)
        covariance = torch.var(X[support], correction=0).reshape(1, 1)
    else:
        support = torch.ones(n_samples, dtype=torch.bool, device=X.device)
        covariance = torch.var(X, correction=0).reshape(1, 1)
        location = torch.mean(X)

    location = location.reshape(1)
    dist, _ = _mahalanobis_dist_sq_features(
        X.reshape(-1, 1),
        location,
        covariance,
    )
    return location, covariance, support, dist


def _c_step_torch(
    X,
    n_support,
    *,
    random_state,
    remaining_iterations=30,
    initial_estimates=None,
):
    n_samples = X.shape[0]
    dist = torch.full(
        (n_samples,),
        float("inf"),
        dtype=X.dtype,
        device=X.device,
    )
    if initial_estimates is None:
        perm = torch.as_tensor(
            random_state.permutation(n_samples),
            device=X.device,
        )
        support = torch.zeros(n_samples, dtype=torch.bool, device=X.device)
        support[perm[:n_support]] = True
    else:
        dist, precision = _mahalanobis_dist_sq_features(X, *initial_estimates)
        support = _support_from_dist_torch(dist, n_support)

    X_support = X[support]
    location = torch.mean(X_support, dim=0)
    covariance = _empirical_covariance_features(
        X_support,
        assume_centered=False,
    )

    det = _fast_logdet_backend(covariance)
    if np.isinf(det):
        precision = torch.linalg.pinv(covariance, hermitian=True)

    previous_det = np.inf
    while (
        det < previous_det
        and remaining_iterations > 0
        and not np.isinf(det)
    ):
        previous_location = location.clone()
        previous_covariance = covariance.clone()
        previous_det = det
        previous_support = support.clone()

        dist, precision = _mahalanobis_dist_sq_features(
            X,
            location,
            covariance,
        )
        support = torch.zeros(n_samples, dtype=torch.bool, device=X.device)
        support[torch.argsort(dist)[:n_support]] = True
        X_support = X[support]
        location = torch.mean(X_support, dim=0)
        covariance = _empirical_covariance_features(
            X_support,
            assume_centered=False,
        )
        det = _fast_logdet_backend(covariance)
        remaining_iterations -= 1

    previous_dist = dist
    dist = torch.sum(((X - location) @ precision) * (X - location), dim=1)
    if np.isinf(det):
        results = location, covariance, det, support, dist
    elif np.allclose(det, previous_det):
        results = location, covariance, det, support, dist
    elif det > previous_det:
        warnings.warn(
            "Determinant has increased; this should not happen: "
            "log(det) > log(previous_det) (%.15f > %.15f). "
            "You may want to try with a higher value of "
            "support_fraction (current value: %.3f)."
            % (det, previous_det, n_support / n_samples),
            RuntimeWarning,
            stacklevel=2,
        )
        results = (
            previous_location,
            previous_covariance,
            previous_det,
            previous_support,
            previous_dist,
        )
    else:
        results = location, covariance, det, support, dist

    if remaining_iterations == 0:
        results = location, covariance, det, support, dist

    return results


def _select_candidates_torch(
    X,
    n_support,
    n_trials,
    *,
    select=1,
    n_iter=30,
    random_state=None,
):
    random_state = check_random_state(random_state)

    if isinstance(n_trials, int):
        run_from_estimates = False
    elif isinstance(n_trials, tuple):
        run_from_estimates = True
        estimates_list = n_trials
        n_trials = estimates_list[0].shape[0]
    else:
        raise TypeError(
            "Invalid 'n_trials' parameter, expected tuple or integer, got "
            f"{n_trials} ({type(n_trials)})"
        )

    all_estimates = [
        _c_step_torch(
            X,
            n_support,
            remaining_iterations=n_iter,
            initial_estimates=None if not run_from_estimates else (
                estimates_list[0][j],
                estimates_list[1][j],
            ),
            random_state=random_state,
        )
        for j in range(n_trials)
    ]

    all_locs, all_covs, all_dets, all_supports, all_ds = zip(*all_estimates)
    index_best = np.argsort(all_dets)[:select]
    best_locations = torch.stack([all_locs[i] for i in index_best], dim=0)
    best_covariances = torch.stack([all_covs[i] for i in index_best], dim=0)
    best_supports = torch.stack([all_supports[i] for i in index_best], dim=0)
    best_ds = torch.stack([all_ds[i] for i in index_best], dim=0)
    return best_locations, best_covariances, best_supports, best_ds


def _fast_mcd_torch(X, *, support_fraction=None, random_state=None):
    X = X.T
    random_state = check_random_state(random_state)
    n_samples, n_features = X.shape

    if n_samples < 2:
        raise ValueError(
            "Minimum Covariance Determinant requires at least 2 samples."
        )

    if support_fraction is None:
        n_support = int(np.ceil(0.5 * (n_samples + n_features + 1)))
    else:
        n_support = int(support_fraction * n_samples)

    if n_features == 1:
        return _mcd_1d_torch(X, n_support)

    if n_samples > 500:
        n_subsets = n_samples // 300
        n_samples_subsets = n_samples // n_subsets
        samples_shuffle = torch.as_tensor(
            random_state.permutation(n_samples),
            device=X.device,
        )
        h_subset = int(
            np.ceil(n_samples_subsets * (n_support / float(n_samples)))
        )
        n_trials_tot = 500
        n_best_sub = 10
        n_trials = max(10, n_trials_tot // n_subsets)
        all_best_locations = []
        all_best_covariances = []
        for i in range(n_subsets):
            low_bound = i * n_samples_subsets
            high_bound = low_bound + n_samples_subsets
            current_subset = X[samples_shuffle[low_bound:high_bound]]
            best_locs, best_covs, _, _ = _select_candidates_torch(
                current_subset,
                h_subset,
                n_trials,
                select=n_best_sub,
                n_iter=2,
                random_state=random_state,
            )
            all_best_locations.extend(list(best_locs))
            all_best_covariances.extend(list(best_covs))

        n_samples_merged = min(1500, n_samples)
        h_merged = int(
            np.ceil(n_samples_merged * (n_support / float(n_samples)))
        )
        n_best_merged = 10 if n_samples > 1500 else 1
        selection = torch.as_tensor(
            random_state.permutation(n_samples)[:n_samples_merged],
            device=X.device,
        )
        locations_merged, covariances_merged, supports_merged, d = (
            _select_candidates_torch(
                X[selection],
                h_merged,
                n_trials=(
                    torch.stack(all_best_locations, dim=0),
                    torch.stack(all_best_covariances, dim=0),
                ),
                select=n_best_merged,
                random_state=random_state,
            )
        )
        if n_samples < 1500:
            location = locations_merged[0]
            covariance = covariances_merged[0]
            support = torch.zeros(n_samples, dtype=torch.bool, device=X.device)
            dist = torch.zeros(n_samples, dtype=X.dtype, device=X.device)
            support[selection] = supports_merged[0]
            dist[selection] = d[0]
        else:
            locations_full, covariances_full, supports_full, d = (
                _select_candidates_torch(
                    X,
                    n_support,
                    n_trials=(locations_merged, covariances_merged),
                    select=1,
                    random_state=random_state,
                )
            )
            location = locations_full[0]
            covariance = covariances_full[0]
            support = supports_full[0]
            dist = d[0]
    else:
        locations_best, covariances_best, _, _ = _select_candidates_torch(
            X,
            n_support,
            n_trials=30,
            select=10,
            n_iter=2,
            random_state=random_state,
        )
        locations_full, covariances_full, supports_full, d = (
            _select_candidates_torch(
                X,
                n_support,
                n_trials=(locations_best, covariances_best),
                select=1,
                random_state=random_state,
            )
        )
        location = locations_full[0]
        covariance = covariances_full[0]
        support = supports_full[0]
        dist = d[0]

    return location, covariance, support, dist


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
        backend = resolve_backend(X)
        iscomplex = not is_real_type(X)
        if iscomplex:
            n_channels, _ = X.shape
            X = backend.concatenate(
                (backend.real(X), backend.imag(X)),
                axis=0,
            )
        cov = func(X, **kwds)
        if iscomplex:
            cov = _make_complex(
                cov[:n_channels, :n_channels] + cov[n_channels:, n_channels:],
                cov[n_channels:, :n_channels] - cov[:n_channels, n_channels:],
                like=X,
            )
        return cov
    return wrapper


@_complex_estimator
def _lwf(X, **kwds):
    """Wrapper for sklearn ledoit wolf covariance estimator"""
    if isinstance(X, np.ndarray):
        C, _ = ledoit_wolf(X.T, **kwds)
        return C
    return _ledoit_wolf_torch(X, **kwds)[0]


@_complex_estimator
def _mcd(X, **kwds):
    """Wrapper for sklearn mcd covariance estimator"""
    if isinstance(X, np.ndarray):
        _, C, _, _ = fast_mcd(X.T, **kwds)
        return C
    return _fast_mcd_torch(X, **kwds)[1]


@_complex_estimator
def _oas(X, **kwds):
    """Wrapper for sklearn oas covariance estimator"""
    if isinstance(X, np.ndarray):
        C, _ = oas(X.T, **kwds)
        return C
    return _oas_torch(X, **kwds)[0]


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
    backend = resolve_backend(X, init)
    n_channels, n_times = X.shape

    if m_estimator == "hub":
        if not 0 < q <= 1:
            raise ValueError(f"Value q must be included in (0, 1] (Got {q})")

        def weight_func(x):  # Example 1, Section V-C in [1]
            c2 = chi2.ppf(q, n_channels) / 2
            b = chi2.cdf(2 * c2, n_channels + 1) + c2 * (1 - q) / n_channels
            return backend.minimum(
                backend.ones(x.shape, like=x, dtype=backend.real_dtype(x)),
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
        X = X - backend.mean(X, axis=1)[:, np.newaxis]
    if init is None:
        cov = X @ ctranspose(X, backend=backend) / n_times
    else:
        cov = check_init(init, n_channels, backend=backend, like=X)

    for _ in range(n_iter_max):

        dist2 = distance_mahalanobis(X, cov, squared=True, backend=backend)
        Xw = backend.sqrt(weight_func(dist2))[np.newaxis, :] * X
        cov_new = Xw @ ctranspose(Xw, backend=backend) / n_times

        norm_delta = backend.as_float(backend.norm_fro(cov_new - cov))
        norm_cov = backend.as_float(backend.norm_fro(cov))
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
    backend = resolve_backend(X)
    if not is_real_type(X):
        n_channels, _ = X.shape
        X_ri = backend.concatenate(
            (backend.real(X), backend.imag(X)),
            axis=0,
        )
        cov = covariance_sch(X_ri)
        return (
            cov[:n_channels, :n_channels]
            + cov[n_channels:, n_channels:]
            + 1j * (
                cov[n_channels:, :n_channels]
                - cov[:n_channels, n_channels:]
            )
        )

    _, n_times = X.shape
    X_c = X - backend.mean(X, axis=1)[:, np.newaxis]
    C_scm = X_c @ ctranspose(X_c, backend=backend) / n_times

    # Compute optimal gamma, the weigthing between SCM and shrinkage estimator
    std = (
        X.std(axis=1)
        if isinstance(X, np.ndarray)
        else torch.std(X, dim=1, correction=0)
    )
    R = n_times / ((n_times - 1.) * backend.outer(std, std))
    R *= C_scm
    var_R = (X_c ** 2) @ ctranspose(X_c ** 2, backend=backend)
    var_R -= 2 * C_scm * (X_c @ ctranspose(X_c, backend=backend))
    var_R += n_times * C_scm ** 2
    var = (
        X.var(axis=1)
        if isinstance(X, np.ndarray)
        else torch.var(X, dim=1, correction=0)
    )
    Xvar = backend.outer(var, var)
    var_R *= n_times / ((n_times - 1) ** 3 * Xvar)
    diag0, diag1 = backend.diag_indices(R.shape[-1], like=R)
    R[diag0, diag1] = 0
    var_R[diag0, diag1] = 0
    denom = backend.as_float(backend.sum(R ** 2))
    gamma = 0 if denom == 0 else max(
        0,
        min(
            1,
            backend.as_float(backend.sum(var_R)) / denom,
        ),
    )

    sigma = (1. - gamma) * (n_times / (n_times - 1.)) * C_scm
    shrinkage = (
        gamma
        * (n_times / (n_times - 1.))
        * backend.diag_embed(backend.diagonal(C_scm))
    )
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
    backend = resolve_backend(X)
    _, n_times = X.shape
    if not assume_centered:
        X = X - backend.mean(X, axis=1)[:, np.newaxis]
    return X @ ctranspose(X, backend=backend) / n_times


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
    backend = resolve_backend(X)
    covmats = backend.zeros((n_matrices, n_channels, n_channels), like=X)
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
    backend = resolve_backend(X, P)
    n_channels_proto, n_times_p = P.shape
    if n_times_p != n_times:
        raise ValueError(
            f"X and P do not have the same n_times: {n_times} and {n_times_p}")
    covmats = backend.zeros(
        (
            n_matrices,
            n_channels + n_channels_proto,
            n_channels + n_channels_proto,
        ),
        like=X,
    )
    for i in range(n_matrices):
        covmats[i] = est(
            backend.concatenate((P, X[i]), axis=0),
            **kwds,
        )
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
    backend = resolve_backend(X)

    Hchannels = backend.eye(n_channels, like=X) - backend.outer(
        backend.ones(n_channels, like=X, dtype=backend.real_dtype(X)),
        backend.ones(n_channels, like=X, dtype=backend.real_dtype(X)),
    ) / n_channels
    Htimes = backend.eye(n_times, like=X) - backend.outer(
        backend.ones(n_times, like=X, dtype=backend.real_dtype(X)),
        backend.ones(n_times, like=X, dtype=backend.real_dtype(X)),
    ) / n_times
    X = Hchannels @ X @ Htimes  # Eq(8), double centering

    covmats = backend.zeros(
        (n_matrices, n_channels + n_times, n_channels + n_times),
        like=X,
    )
    for i in range(n_matrices):
        Y = backend.concatenate((
            backend.concatenate(
                (X[i], alpha * backend.eye(n_channels, like=X)),
                axis=1,
            ),
            backend.concatenate(
                (
                    alpha * backend.eye(n_times, like=X),
                    backend.swapaxes(X[i], -2, -1),
                ),
                axis=1,
            ),
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
    backend = resolve_backend(X)

    if sum(blocks) != n_channels:
        raise ValueError("Sum of individual block sizes "
                         "must match number of channels of X.")

    covmats = backend.zeros((n_matrices, n_channels, n_channels), like=X)
    for i in range(n_matrices):
        idx_start = 0
        for j in blocks:
            covmats[i, idx_start:idx_start+j, idx_start:idx_start+j] = est(
                X[i, idx_start:idx_start+j, :],
                **kwds,
            )
            idx_start += j

    return covmats


###############################################################################


def eegtocov(sig, window=128, overlapp=0.5, padding=True, estimator="cov"):
    """Convert EEG signal to covariance using sliding window."""
    est = check_function(estimator, cov_est_functions)
    backend = resolve_backend(sig)
    X = []
    if padding:
        padd = backend.zeros((int(window / 2), sig.shape[1]), like=sig)
        sig = backend.concatenate((padd, sig, padd), axis=0)

    n_times, n_channels = sig.shape
    jump = int(window * overlapp)
    ix = 0
    while (ix + window < n_times):
        X.append(est(sig[ix:ix + window, :].T))
        ix = ix + jump

    return backend.stack(X, axis=0)


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

    backend = resolve_backend(X)
    n_channels, n_times = X.shape
    n_freqs = int(window / 2) + 1  # X real signal => compute half-spectrum
    step = int((1.0 - overlap) * window)
    n_windows = int((n_times - window) / step + 1)
    if isinstance(X, np.ndarray):
        win = np.hanning(window)

        # FFT calculation on all windows
        shape = (n_channels, n_windows, window)
        strides = X.strides[:-1] + (step * X.strides[-1], X.strides[-1])
        Xs = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        fdata = np.fft.rfft(Xs * win, n=window).transpose(1, 0, 2)
    else:
        X = X.contiguous()
        win = torch.hann_window(
            window,
            periodic=False,
            dtype=backend.real_dtype(X),
            device=X.device,
        )
        shape = (n_channels, n_windows, window)
        strides = (X.stride(0), step * X.stride(1), X.stride(1))
        Xs = X.as_strided(shape, strides)
        fdata = torch.fft.rfft(Xs * win, n=window).permute(1, 0, 2)

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
                dtype=backend.real_dtype(X),
            ) * float(fs / window)
        fix = (f >= fmin) & (f <= fmax)
        fdata = fdata[:, :, fix]
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

    n_freqs = fdata.shape[2]
    S = backend.zeros((n_channels, n_channels, n_freqs), like=fdata)
    for i in range(n_freqs):
        spec = fdata[:, :, i]
        S[:, :, i] = ctranspose(spec, backend=backend) @ spec
    S /= n_windows * backend.sum(win ** 2)

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

    return resolve_backend(S).real(S), freqs


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
    C : ndarray, shape (n_channels, n_channels, n_freqs)
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
    backend = resolve_backend(S)
    S2 = backend.abs(S) ** 2  # squared cross-spectral modulus

    C = backend.zeros_like(S2)
    f_inds = list(range(C.shape[-1]))

    # lagged coh not defined for DC and Nyquist bins, because S is real
    if coh == "lagged":
        if freqs is None:
            f_inds = list(range(1, C.shape[-1] - 1))
            warnings.warn(
                "DC and Nyquist bins are not defined for lagged-"
                "coherence: filled with zeros",
                stacklevel=2,
            )
        else:
            if isinstance(freqs, np.ndarray):
                f_inds_ = np.where((freqs > 0) & (freqs < fs / 2))[0].tolist()
            else:
                f_inds_ = torch.nonzero(
                    (freqs > 0) & (freqs < fs / 2),
                    as_tuple=False,
                ).flatten().tolist()
            if f_inds_ != f_inds:
                warnings.warn(
                    "DC and Nyquist bins are not defined for lagged-"
                    "coherence: filled with zeros",
                    stacklevel=2,
                )
            f_inds = f_inds_

    for f in f_inds:
        psd = backend.sqrt(backend.diagonal(S2[..., f]))
        psd_prod = backend.outer(psd, psd)
        if coh == "ordinary":
            C[..., f] = S2[..., f] / psd_prod
        elif coh == "instantaneous":
            C[..., f] = backend.real(S[..., f]) ** 2 / psd_prod
        elif coh == "lagged":
            S_real = backend.real(S[..., f])
            S_real_copy = backend.zeros_like(S_real)
            S_real_copy[...] = S_real
            diag0, diag1 = backend.diag_indices(S_real.shape[-1], like=S_real)
            S_real_copy[diag0, diag1] = 0.0
            denom = backend.maximum(psd_prod - S_real_copy ** 2, 1e-10)
            C[..., f] = backend.imag(S[..., f]) ** 2 / denom
        elif coh == "imaginary":
            C[..., f] = backend.imag(S[..., f]) ** 2 / psd_prod
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
    backend = resolve_backend(X)
    if not is_square(X):
        raise ValueError("Matrices must be square")

    if norm == "corr":
        stddev = backend.sqrt(backend.abs(backend.diagonal(X)))
        denom = stddev[..., :, np.newaxis] * stddev[..., np.newaxis, :]
    elif norm == "trace":
        denom = backend.sum(backend.diagonal(X), axis=-1)
    elif norm == "determinant":
        _, logabsdet = backend.slogdet(X)
        denom = backend.exp(logabsdet / X.shape[-1])
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
    backend = resolve_backend(X)
    if not is_square(X):
        raise ValueError("Matrices must be square")

    X2 = X**2
    # sum of squared diagonal elements
    denom = backend.sum(backend.diagonal(X2), axis=-1)
    # sum of squared off-diagonal elements
    num = backend.sum(X2, axis=(-2, -1)) - denom
    weights = (1.0 / (X.shape[-1] - 1)) * (num / denom)
    return weights
