import numpy as np

from scipy.linalg import sqrtm, inv
from numpy import iscomplexobj, real, any, isfinite
from sklearn.base import BaseEstimator, TransformerMixin
from pyriemann.estimation import covariances

from pyriemann.utils.mean import mean_covariance
from pyriemann.transfer import decode_domains


def compute_ref_euclidean(data):
    '''
    Calculate the reference matrix for computing the Euclidean alignment,

    Parameters
    ----------
    data: numpy array.
    It can be either raw trials with shape (n, c, t) or covariances (n, c, c)
    Reference trials to compute the alignment matrix

    Returns
    -------
    r_ea: numpy array with shape (c, c)
    Reference matrix

    '''

    mean = mean_covariance(data, metric='euclid')

    compare = np.allclose(mean, np.identity(mean.shape[0]))

    if not compare:
        if iscomplexobj(mean):
            print("covariance matrix problem")
        if iscomplexobj(sqrtm(mean)):
            print("covariance matrix problem sqrt")

        r_ea = inv(sqrtm(mean))

        if iscomplexobj(r_ea):
            print("WARNING! Covariance matrix was not SPD somehow. " +
                  "Can be caused by running ICA-EOG rejection, if " +
                  "not, check data!!")
            r_ea = real(r_ea).astype(np.float64)
        elif not any(isfinite(r_ea)):
            print("WARNING! Not finite values in R Matrix")

    else:
        print("Already aligned!")
        r_ea = mean

    return r_ea


class TransformEA(BaseEstimator, TransformerMixin):
    """Euclidean Alignment, based on [1].

    The Euclidean Alignment is a transformation that recenter each domain's
    Euclidean mean covariance matrix to the Identity, making distributions
    more similar.

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
    _ref : list
        Contains the reference matrix for each domain.
        Order of the list is the same as 'domains'

    _cov : Dict
        Dictionary containing the covariance matrices for each domain.
        Order of the list is the same as 'domains'

    References
    ----------
    .. [1] `Transfer Learning for Brain-Computer Interfaces:
        A Euclidean Space Data Alignment Approach
        <https://arxiv.org/abs/1808.05464>`_
        He He and Dongrui Wu, IEEE Transactions on Biomedical Engineering, 2019

    """
    def __init__(self, estimator='lwf', dtype='covmat'):
        self.estimator = estimator
        self.dtype = dtype
        self._ref = list()
        self._cov = {}

    def _compute_ref(self, X):
        ref = compute_ref_euclidean(X)

        self._ref.append(ref)

    def _fit_sw(self, X, domains):

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
            ref_d = self._ref[i]

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
        self._cov[d] = cov
        return cov

    def fit(self, X, y_enc):

        self._ref.clear()

        _, _, domains = decode_domains(X, y_enc)

        self._fit_sw(X, domains)

        return self

    def transform(self, X, y_enc=None):
        """
        Used for transforming the target data after fitting the calibration
        """

        # In this case, X is test and you suppose that calibration was previously fitted
        ref = self._ref[0]

        if self.dtype == 'covmat':
            X_align = ref @ X @ ref
        else:
            X_align = ref @ X

        return X_align

    def fit_transform(self, X, y_enc=None, **fit_params):
        """
        Used for fitting and transforming 'train' data.
        """

        self.fit(X, y_enc)

        _, _, domains = decode_domains(X, y_enc)

        X_align = self._transform_sw(X, domains)

        return X_align

    def __sklearn_is_fitted__(self):
        """Return True since Transfomer is stateless."""
        return True
