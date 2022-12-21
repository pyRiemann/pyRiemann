import numbers

import numpy as np
from scipy.linalg import eigh

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import stable_cumsum

from .utils.mean import mean_covariance
from .utils.base import sqrtm, invsqrtm


class Whitening(BaseEstimator, TransformerMixin):
    """Whitening, and optional unsupervised dimension reduction.

    Implementation of the whitening, and an optional unsupervised dimension
    reduction, with SPD matrices as inputs.

    Parameters
    ----------
    metric : str, default='euclid'
        The metric for the estimation of mean matrix used for whitening and
        dimension reduction.
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
    filters_ : ndarray, shape ``(n_channels_, n_components_)``
        If fit, the spatial filters to whiten SPD matrices.
    inv_filters_ : ndarray, shape ``(n_components_, n_channels_)``
        If fit, the spatial filters to unwhiten SPD matrices.

    Notes
    -----
    .. versionadded:: 0.2.7

    """

    def __init__(self, metric='euclid', dim_red=None, verbose=False):
        """Init."""
        self.metric = metric
        self.dim_red = dim_red
        self.verbose = verbose

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
        # weighted mean of input SPD matrices
        Xm = mean_covariance(
            X,
            metric=self.metric,
            sample_weight=sample_weight
        )

        # whitening without dimension reduction
        if self.dim_red is None:
            self.n_components_ = X.shape[-1]
            self.filters_ = invsqrtm(Xm)
            self.inv_filters_ = sqrtm(Xm)

        # whitening with dimension reduction
        elif isinstance(self.dim_red, dict):

            eigvals, eigvecs = self._prepare_dimension_reduction(X, Xm)

            # dimension reduction
            if self.verbose:
                print('Dimension reduction of Whitening on %d components'
                      % self.n_components_)
            pca_filters = eigvecs[:, :self.n_components_]
            pca_sqrtvals = np.sqrt(eigvals[:self.n_components_])
            # whitening
            self.filters_ = pca_filters @ np.diag(1. / pca_sqrtvals)
            self.inv_filters_ = np.diag(pca_sqrtvals).T @ pca_filters.T

        else:
            raise ValueError('Unknown type for parameter dim_red: %r'
                             % type(self.dim_red))

        return self

    def _prepare_dimension_reduction(self, X, Xm):
        """Prepare dimension reduction."""
        if len(self.dim_red) > 1:
            raise ValueError(
                'Dictionary dim_red must contain only one element (Got %d)'
                % len(self.dim_red))
        dim_red_key = next(iter(self.dim_red))
        dim_red_val = self.dim_red.get(dim_red_key)

        eigvals, eigvecs = eigh(Xm, eigvals_only=False)
        eigvals = eigvals[::-1]       # sort eigvals in descending order
        eigvecs = np.fliplr(eigvecs)  # idem for eigvecs

        if dim_red_key == 'n_components':
            if dim_red_val < 1:
                raise ValueError(
                    'Value n_components must be superior to 1 (Got %d)'
                    % dim_red_val)
            if not isinstance(dim_red_val, numbers.Integral):
                raise ValueError(
                    'n_components=%d must be of type int (Got %r)'
                    % (dim_red_val, type(dim_red_val)))
            self.n_components_ = min(dim_red_val, X.shape[-1])

        elif dim_red_key == 'expl_var':
            if not 0 < dim_red_val <= 1:
                raise ValueError(
                    'Value expl_var must be included in (0, 1] (Got %d)'
                    % dim_red_val)
            cum_expl_var = stable_cumsum(eigvals / eigvals.sum())
            if self.verbose:
                print('Cumulative explained variance: \n %r'
                      % cum_expl_var)
            self.n_components_ = np.searchsorted(
                cum_expl_var, dim_red_val, side='right') + 1

        elif dim_red_key == 'max_cond':
            if dim_red_val <= 1:
                raise ValueError(
                    'Value max_cond must be strictly superior to 1 '
                    '(Got %d)' % dim_red_val)
            conds = eigvals[0] / eigvals
            if self.verbose:
                print('Condition numbers: \n %r' % conds)
            self.n_components_ = np.searchsorted(
                conds, dim_red_val, side='left')

        else:
            raise ValueError(
                'Unknown key in parameter dim_red: %r' % dim_red_key)

        return eigvals, eigvecs

    def transform(self, X):
        """Apply whitening spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        Xw : ndarray, shape (n_matrices, n_components, n_components)
            Set of whitened, and optionally reduced, SPD matrices.
        """
        Xw = self.filters_.T @ X @ self.filters_
        return Xw

    def inverse_transform(self, X):
        """Apply inverse whitening spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_components, n_components)
            Set of whitened, and optionally reduced, SPD matrices.

        Returns
        -------
        Xiw : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of unwhitened, and optionally unreduced, SPD matrices.
        """
        Xiw = self.inv_filters_.T @ X @ self.inv_filters_
        return Xiw
