import numpy as np

from warnings import warn

from scipy.linalg import sqrtm, inv
from numpy import iscomplexobj, real, any, isfinite
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import covariances

from ..utils.mean import mean_covariance
from ..transfer import decode_domains


def _compute_ref_euclidean(data):

    mean = mean_covariance(data, metric='euclid')

    compare = np.allclose(mean, np.identity(mean.shape[0]))

    if not compare:
        if iscomplexobj(mean):
            warn("Covariance matrix problem")
        if iscomplexobj(sqrtm(mean)):
            warn("Covariance matrix problem sqrt")

        ref_ea = inv(sqrtm(mean))

        if iscomplexobj(ref_ea):
            warn("Covariance matrix was not SPD somehow. " +
                          "Can be caused by running ICA-EOG rejection, if " +
                          "not, check data!!")
            ref_ea = real(ref_ea).astype(np.float64)
        elif not any(isfinite(ref_ea)):
            warn("Not finite values in R Matrix")

    else:
        warn("Already aligned!")
        ref_ea = mean

    return ref_ea


class EuclideanAlignment(BaseEstimator, TransformerMixin):
    r"""Euclidean Alignment, based on [1],[2].

    The Euclidean Alignment is a transformation that recenter each domain's
    Euclidean mean covariance matrix to the Identity, making distributions
    more similar.

    .. math::
    \mathbf{M} = \frac{1}{n} \sum_i \ \mathbf{X}_i

    This transformation accept both raw signals and covariance matrices as
    data inputs. For raw trials, you can determine the estimator using the
    'estimtor' parameter.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training dataset (target and source) and .transform() on the test
       partition of the target dataset.

    Parameters
    ----------
    dtype: str
        Type of input data, it can be 'covmat' for covariance or 'raw' for raw trials

    estimator: str
        If dtype=='raw', it represents the covariance estimator

    Attributes
    ----------
    ref_ : list
        Contains the reference matrix for each domain.
        Order of the list is the same as 'domains'

    cov_ : Dict
        Dictionary containing the covariance matrices for each domain.
        Order of the list is the same as 'domains'

    References
    ----------
    .. [1] `Transfer Learning for Brain-Computer Interfaces:
        A Euclidean Space Data Alignment Approach
        <https://arxiv.org/abs/1808.05464>`_
        He He and Dongrui Wu, IEEE Transactions on Biomedical Engineering, 2019

    .. [2] 'A Systematic Evaluation of Euclidean Alignment
        with Deep Learning for EEG Decoding
        <https://arxiv.org/abs/2401.10746>`
        B Junqueira, B Aristimunha, S Chevallier, and R Y. de Camargo

    Notes
    -------
    ..versionadded:: 0.6.0

    """

    _dtypes = ['covmat', 'raw']

    def __init__(self, estimator='lwf', dtype='covmat'):
        """Init"""

        self.estimator = estimator
        self.dtype = dtype
        if self.dtype not in self._dtypes:
            raise ValueError(f"Unknown data type '{self._dtype}'. Must be one of "
                             f"{self._dtypes}")
        self.ref_ = []
        self.cov_ = {}

    def _compute_ref(self, X):
        """
        Compute reference matrix
        """

        ref = _compute_ref_euclidean(X)

        self.ref_.append(ref)

    def _fit_sw(self, X, domains):
        """
        Fit data in a subject-wise/domain-wise way

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_windows) if raw trials
        or (n_trials, n_channels, n_channels) if covariance matrices
            Set of SPD matrices.
        domains : ndarray, shape (n_trials,)
            Domain of each trial

        """

        for i in range(len(np.unique(domains))):

            d = np.unique(domains)[i]

            X_d = X[domains == d]

            if self.dtype != 'covmat':
                X_d = self._transform_cov(X_d, d)

            self._compute_ref(X_d)

    def _transform_sw(self, X, domains):
        """Align each domain"""

        X_align = []
        for i in range(len(np.unique(domains))):

            d = np.unique(domains)[i]
            X_d = X[domains == d]
            ref_d = self.ref_[i]

            if self.dtype == 'covmat':
                align = ref_d @ X_d @ ref_d
            else:
                align = ref_d @ X_d
            X_align.append(align)

        X_align = np.concatenate(X_align)
        return X_align

    def _transform_cov(self, X, d):
        """ Compute covariance matrices of trials in array X and save"""

        cov = covariances(X, estimator=self.estimator)
        self.cov_[d] = cov
        return cov

    def fit(self, X, y_enc):
        """Fit (estimates) the reference matrix for each domain.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels) or (n_trials, n_channels, n_windows)
            Data trials, can be SPD matrices or raw.
        y_enc : ndarray, shape (n_trials,)
            Extended labels for each trial. Encodes domain information.

        Returns
        -------
        self : EuclideanAlignment instance
            The EuclideanAlignment instance.
        """

        self.ref_.clear()

        _, _, domains = decode_domains(X, y_enc)

        self._fit_sw(X, domains)

        return self

    def transform(self, X, y_enc=None):
        """Transforms target data points after calibration step.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels) or (n_trials, n_channels, n_windows)
            Target trials, can be SPD matrices or raw.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X : ndarray, shape (n_trials, n_channels, n_channels) or (n_trials, n_channels, n_windows)
            Target trials with Euclidean mean at Identity

        """

        # In this case, X is test and you suppose that calibration was previously fitted
        ref = self.ref_[0]

        if self.dtype == 'covmat':
            X_align = ref @ X @ ref
        else:
            X_align = ref @ X

        return X_align

    def fit_transform(self, X, y_enc=None, **fit_params):
        """Fit EuclideanAlignment and then transform data points.

        Calculate the mean of all matrices in each domain and then recenter
        them to Identity.

        .. note::
           This method is designed for using at training time. The output for
           .fit_transform() will be different than using .fit() and
           .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels) or (n_trials, n_channels, n_windows)
            Target trials, can be SPD matrices or raw.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each trial. Encodes domain information.

        Returns
        -------
        X : ndarray, shape (n_trials, n_channels, n_channels) or (n_trials, n_channels, n_windows)
            Target trials with Euclidean mean at Identity

        """

        self.fit(X, y_enc)

        _, _, domains = decode_domains(X, y_enc)

        X_align = self._transform_sw(X, domains)

        return X_align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True
