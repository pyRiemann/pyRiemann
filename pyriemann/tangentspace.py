"""Tangent space functions."""
from .utils.mean import mean_covariance
from .utils.tangentspace import tangent_space, untangent_space
from .utils.base import check_version

import numpy
from sklearn.base import BaseEstimator, TransformerMixin
if check_version('sklearn', '0.17'):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
else:
    from sklearn.lda import LDA


class TangentSpace(BaseEstimator, TransformerMixin):

    """Tangent space project TransformerMixin.

    Tangent space projection map a set of covariance matrices to their
    tangent space according to [1]. The Tangent space projection can be
    seen as a kernel operation, cf [2]. After projection, each matrix is
    represented as a vector of size :math:`N(N+1)/2` where N is the
    dimension of the covariance matrices.

    Tangent space projection is useful to convert covariance matrices in
    euclidean vectors while conserving the inner structure of the manifold.
    After projection, standard processing and vector-based classification can
    be applied.

    Tangent space projection is a local approximation of the manifold. it takes
    one parameter, the reference point, that is usually estimated using the
    geometric mean of the covariance matrices set you project. if the function
    `fit` is not called, the identity matrix will be used as reference point.
    This can lead to serious degradation of performances.
    The approximation will be bigger if the matrices in the set are scattered
    in the manifold, and lower if they are grouped in a small region of the
    manifold.

    After projection, it is possible to go back in the manifold using the
    inverse transform.

    Parameters
    ----------
    metric : string (default: 'riemann')
        The type of metric used for reference point mean estimation.
        see `mean_covariance` for the list of supported metric.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.


    Attributes
    ----------
    reference_ : ndarray
        If fit, the reference point for tangent space mapping.

    See Also
    --------
    FgMDM
    FGDA

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass
    Brain-Computer Interface Classification by Riemannian Geometry,"" in IEEE
    Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self, metric='riemann', tsupdate=False):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate

    def fit(self, X, y=None, sample_weight=None):
        """Fit (estimates) the reference point.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.

        Returns
        -------
        self : TangentSpace instance
            The TangentSpace instance.
        """
        # compute mean covariance
        self.reference_ = mean_covariance(X, metric=self.metric,
                                          sample_weight=sample_weight)
        return self

    def _check_data_dim(self, X):
        """Check data shape and return the size of cov mat."""
        shape_X = X.shape
        if len(X.shape) == 2:
            Ne = (numpy.sqrt(1 + 8 * shape_X[1]) - 1) / 2
            if Ne != int(Ne):
                raise ValueError("Shape of Tangent space vector does not"
                                 " correspond to a square matrix.")
            return int(Ne)
        elif len(X.shape) == 3:
            if shape_X[1] != shape_X[2]:
                raise ValueError("Matrices must be square")
            return int(shape_X[1])
        else:
            raise ValueError("Shape must be of len 2 or 3.")

    def _check_reference_points(self, X):
        """Check reference point status, and force it to identity if not."""
        if not hasattr(self, 'reference_'):
            self.reference_ = numpy.eye(self._check_data_dim(X))
        else:
            shape_cr = self.reference_.shape[0]
            shape_X = self._check_data_dim(X)

            if shape_cr != shape_X:
                raise ValueError('Data must be same size of reference point.')

    def transform(self, X):
        """Tangent space projection.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        ts : ndarray, shape (n_trials, n_ts)
            the tangent space projection of the matrices.
        """
        self._check_reference_points(X)
        if self.tsupdate:
            Cr = mean_covariance(X, metric=self.metric)
        else:
            Cr = self.reference_
        return tangent_space(X, Cr)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.

        Returns
        -------
        ts : ndarray, shape (n_trials, n_ts)
            the tangent space projection of the matrices.
        """
        # compute mean covariance
        self._check_reference_points(X)
        self.reference_ = mean_covariance(X, metric=self.metric,
                                          sample_weight=sample_weight)
        return tangent_space(X, self.reference_)

    def inverse_transform(self, X, y=None):
        """Inverse transform.

        Project back a set of tangent space vector in the manifold.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_ts)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        cov : ndarray, shape (n_trials, n_channels, n_channels)
            the covariance matrices corresponding to each of tangent vector.
        """
        self._check_reference_points(X)
        return untangent_space(X, self.reference_)


class FGDA(BaseEstimator, TransformerMixin):

    """Fisher Geodesic Discriminant analysis.

    Project data in Tangent space, apply a FLDA to reduce dimention, and
    project filtered data back in the manifold.
    For a complete description of the algorithm, see [1]

    Parameters
    ----------
    metric : string (default: 'riemann')
        The type of metric used for reference point mean estimation.
        see `mean_covariance` for the list of supported metric.
    tsupdate : bool (default False)
        Activate tangent space update for covariante shift correction between
        training and test, as described in [2]. This is not compatible with
        online implementation. Performance are better when the number of trials
        for prediction is higher.

    See Also
    --------
    FgMDM
    TangentSpace

    References
    ----------
    [1] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Riemannian geometry
    applied to BCI classification", 9th International Conference Latent
    Variable Analysis and Signal Separation (LVA/ICA 2010), LNCS vol. 6365,
    2010, p. 629-636.

    [2] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of
    covariance matrices using a Riemannian-based kernel for BCI applications",
    in NeuroComputing, vol. 112, p. 172-178, 2013.
    """

    def __init__(self, metric='riemann', tsupdate=False):
        """Init."""
        self.metric = metric
        self.tsupdate = tsupdate

    def _fit_lda(self, X, y, sample_weight=None):
        """Helper to fit LDA."""
        self.classes_ = numpy.unique(y)
        self._lda = LDA(n_components=len(self.classes_) - 1,
                        solver='lsqr',
                        shrinkage='auto')

        ts = self._ts.fit_transform(X, sample_weight=sample_weight)
        self._lda.fit(ts, y)

        W = self._lda.coef_.copy()
        self._W = numpy.dot(
            numpy.dot(W.T, numpy.linalg.pinv(numpy.dot(W, W.T))), W)
        return ts

    def _retro_project(self, ts):
        """Helper to project back in the manifold."""
        ts = numpy.dot(ts, self._W)
        return self._ts.inverse_transform(ts)

    def fit(self, X, y=None, sample_weight=None):
        """Fit (estimates) the reference point and the FLDA.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.

        Returns
        -------
        self : FGDA instance
            The FGDA instance.
        """
        self._ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        self._fit_lda(X, y, sample_weight=sample_weight)
        return self

    def transform(self, X):
        """Filtering operation.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.

        Returns
        -------
        covs : ndarray, shape (n_trials, n_channels, n_channels)
            covariances matrices after filtering.
        """
        ts = self._ts.transform(X)
        return self._retro_project(ts)

    def fit_transform(self, X, y=None, sample_weight=None):
        """Fit and transform in a single function.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default None)
            Not used, here for compatibility with sklearn API.
        sample_weight : ndarray | None (default None)
            weight of each sample.

        Returns
        -------
        covs : ndarray, shape (n_trials, n_channels, n_channels)
            covariances matrices after filtering.
        """
        self._ts = TangentSpace(metric=self.metric, tsupdate=self.tsupdate)
        ts = self._fit_lda(X, y, sample_weight=sample_weight)
        return self._retro_project(ts)
