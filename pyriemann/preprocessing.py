import numbers

import numpy as np
from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import stable_cumsum

from .utils.base import sqrtm, invsqrtm
from .utils.geodesic import geodesic
from .utils.mean import mean_covariance


class Whitening(TransformerMixin, BaseEstimator):
    """Whitening, and optional unsupervised dimension reduction.

    Implementation of the whitening, and an optional unsupervised dimension
    reduction, with SPD matrices as inputs.

    Parameters
    ----------
    metric : str, default="euclid"
        Metric for the estimation of mean matrix used for whitening and
        dimension reduction.
        For the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`.
    dim_red : None | dict, default=None
        If ``None`` :
            no dimension reduction during whitening.
        If ``{'n_components': val}`` :
            dimension reduction defining the number of components;
            ``val`` must be an integer superior to 1.
        If ``{'expl_var': val}`` :
            dimension reduction selecting the number of components such that
            the amount of variance that needs to be explained is greater than
            the percentage specified by ``val``.
            ``val`` must be a float in (0,1], typically ``0.99``.
        If ``{'max_cond': val}`` :
            dimension reduction selecting the number of components such that
            the condition number of the mean matrix is lower than ``val``.
            This threshold has a physiological interpretation, because it can
            be viewed as the ratio between the power of the strongest component
            (usually, eye-blink source) and the power of the lowest component
            you don't want to keep (acquisition sensor noise).
            ``val`` must be a float strictly superior to 1, typically 100.
    verbose : bool, default=False
        Verbose flag.

    Attributes
    ----------
    n_components_ : int
        If fit, the number of components after dimension reduction.
    filters_ : ndarray, shape (n_channels, ``n_components_``)
        If fit, the spatial filters to whiten SPD matrices.
    inv_filters_ : ndarray, shape (``n_components_``, n_channels)
        If fit, the spatial filters to unwhiten SPD matrices.

    Notes
    -----
    .. versionadded:: 0.2.7

    """

    def __init__(self, metric="euclid", dim_red=None, verbose=False):
        """Init."""
        self.metric = metric
        self.dim_red = dim_red
        self.verbose = verbose
        self._n_matrices_cum = 0

    def fit(self, X, y=None, sample_weight=None):
        """Train whitening spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Ignored as unsupervised.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weight of each matrix, to compute the weighted mean matrix used for
            whitening and dimension reduction. If None, it uses equal weights.

        Returns
        -------
        self : Whitening instance
            The Whitening instance.
        """
        # weighted mean of input matrices
        self._mean = mean_covariance(
            X,
            metric=self.metric,
            sample_weight=sample_weight
        )

        # whitening without dimension reduction
        if self.dim_red is None:
            self.n_components_ = X.shape[-1]
            self.filters_ = invsqrtm(self._mean)
            self.inv_filters_ = sqrtm(self._mean)

        # whitening with dimension reduction
        elif isinstance(self.dim_red, dict):
            self._get_eig()
            self._get_n_components(X)
            self._reduce_and_whiten()

        else:
            raise ValueError("Unknown type for parameter dim_red: %r"
                             % type(self.dim_red))

        self._n_matrices_cum = X.shape[0]

        return self

    def _get_eig(self):
        """Compute eigen values and eigen vectors."""
        eigvals, eigvecs = eigh(self._mean, eigvals_only=False)
        self._eigvals = eigvals[::-1]       # sort eigvals in descending order
        self._eigvecs = np.fliplr(eigvecs)  # idem for eigvecs

    def _get_n_components(self, X):
        """Compute the number of components for dimension reduction."""
        if len(self.dim_red) > 1:
            raise ValueError(
                "Dictionary dim_red must contain only one element (Got %d)"
                % len(self.dim_red))
        dim_red_key = next(iter(self.dim_red))
        dim_red_val = self.dim_red.get(dim_red_key)

        if dim_red_key == "n_components":
            if dim_red_val < 1:
                raise ValueError(
                    "Value n_components must be superior to 1 (Got %d)"
                    % dim_red_val)
            if not isinstance(dim_red_val, numbers.Integral):
                raise ValueError(
                    "n_components=%d must be of type int (Got %r)"
                    % (dim_red_val, type(dim_red_val)))
            self.n_components_ = min(dim_red_val, X.shape[-1])

        elif dim_red_key == "expl_var":
            if not 0 < dim_red_val <= 1:
                raise ValueError(
                    "Value expl_var must be included in (0, 1] (Got %d)"
                    % dim_red_val)
            cum_expl_var = stable_cumsum(self._eigvals / self._eigvals.sum())
            if self.verbose:
                print("Cumulative explained variance: \n %r"
                      % cum_expl_var)
            self.n_components_ = np.searchsorted(
                cum_expl_var, dim_red_val, side="right") + 1

        elif dim_red_key == "max_cond":
            if dim_red_val <= 1:
                raise ValueError(
                    "Value max_cond must be strictly superior to 1 "
                    "(Got %d)" % dim_red_val)
            conds = self._eigvals[0] / self._eigvals
            if self.verbose:
                print("Condition numbers: \n %r" % conds)
            self.n_components_ = np.searchsorted(
                conds, dim_red_val, side="left")

        else:
            raise ValueError(
                "Unknown key in parameter dim_red: %r" % dim_red_key)

        if self.verbose:
            print("Dimension reduction of Whitening on %d components"
                  % self.n_components_)

    def _reduce_and_whiten(self):
        """Compute spatial filters to reduce and whiten matrices."""
        # dimension reduction
        pca_filters = self._eigvecs[:, :self.n_components_]
        pca_sqrtvals = np.sqrt(self._eigvals[:self.n_components_])
        # whitening
        self.filters_ = pca_filters * (1. / pca_sqrtvals)[np.newaxis, :]
        self.inv_filters_ = pca_sqrtvals[:, np.newaxis] * pca_filters.T

    def partial_fit(self, X, y=None, *, sample_weight=None, alpha=None):
        """Partially fit whitening spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Ignored as unsupervised.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.
        alpha : float | None, default=None
            Update rate in [0, 1] for the mean: 0 for no update, 1 for full
            update.
            If None, ``alpha`` is defined as ``n_matrices`` divided by the
            number of matrices that have been already used for fit.

        Returns
        -------
        self : Whitening instance
            The Whitening instance.

        Notes
        -----
        .. versionadded:: 0.7
        """
        n_matrices, n_channels, _ = X.shape
        self._n_matrices_cum += n_matrices

        if alpha is None:
            alpha = n_matrices / self._n_matrices_cum
        if not 0 <= alpha <= 1:
            raise ValueError("Parameter alpha must be in [0, 1]")
        if alpha == 0:
            return self

        if not hasattr(self, "_mean"):
            self._mean = mean_covariance(
                X,
                metric=self.metric,
                sample_weight=sample_weight,
            )
            self.n_components_ = n_channels
        elif n_channels != self._mean.shape[-1]:
            raise ValueError(
                "X does not have the good number of channels. Should be %d but"
                " got %d." % (self._mean.shape[-1], n_channels))
        else:
            Xm = mean_covariance(
                X,
                metric=self.metric,
                sample_weight=sample_weight,
            )
            self._mean = geodesic(self._mean, Xm, alpha, metric=self.metric)

        self._get_eig()
        self._reduce_and_whiten()

        return self

    def transform(self, X):
        """Apply whitening spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, ``n_components_``, \
                ``n_components_``)
            Set of whitened, and optionally reduced, SPD matrices.
        """
        return self.filters_.T @ X @ self.filters_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Ignored as unsupervised.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weight of each matrix, to compute the weighted mean matrix used for
            whitening and dimension reduction. If None, it uses equal weights.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, ``n_components_``, \
                ``n_components_``)
            Set of whitened, and optionally reduced, SPD matrices.
        """
        return self.fit(X, y, sample_weight=sample_weight).transform(X)

    def inverse_transform(self, X):
        """Apply inverse whitening spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, ``n_components_``, ``n_components_``)
            Set of whitened, and optionally reduced, SPD matrices.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of unwhitened, and optionally unreduced, SPD matrices.
        """
        return self.inv_filters_.T @ X @ self.inv_filters_
