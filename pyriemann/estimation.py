import numpy

from .spatialfilters import Xdawn
from .utils.covariance import covariances, covariances_EP, cospectrum
from sklearn.base import BaseEstimator, TransformerMixin


###############################################################################


class Covariances(BaseEstimator, TransformerMixin):

    """
    compute the covariances matrices

    """

    def __init__(self, estimator='scm'):
        self.estimator = estimator

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        covmats = covariances(X, estimator=self.estimator)
        return covmats

    def fit_transform(self, X, y=None):
        return self.transform(X)

###############################################################################


class ERPCovariances(BaseEstimator, TransformerMixin):

    """
    Compute special form ERP cov mat

    """

    def __init__(self, classes=None, estimator='scm', svd=None):
        self.classes = classes
        self.estimator = estimator
        self.svd = svd

        if svd is not None:
            if not isinstance(svd,int):
                raise TypeError('svd must be None or int')

    def fit(self, X, y):

        if self.classes is not None:
            classes = self.classes
        else:
            classes = numpy.unique(y)

        self.P = []
        for c in classes:
            # Prototyped responce for each class
            P = numpy.mean(X[y == c, :, :], axis=0)

            # Apply svd if requested
            if self.svd is not None:
                U, s, V = numpy.linalg.svd(P)
                P = numpy.dot(U[:, 0:self.svd].T,P)

            self.P.append(P)

        self.P = numpy.concatenate(self.P, axis=0)
        return self

    def transform(self, X):

        covmats = covariances_EP(X, self.P, estimator=self.estimator)
        return covmats

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

###############################################################################


class XdawnCovariances(BaseEstimator, TransformerMixin):

    """
    Compute xdawn, project the signal and compute the covariances

    """

    def __init__(self, nfilter=4, applyfilters=True, classes=None,
                 estimator='scm'):
        self.Xd = Xdawn(nfilter=nfilter, classes=classes)
        self.applyfilters = applyfilters
        self.estimator = estimator

    def fit(self, X, y):
        self.Xd.fit(X, y)
        return self

    def transform(self, X):
        if self.applyfilters:
            X = self.Xd.transform(X)

        covmats = covariances_EP(X, self.Xd.evokeds_, estimator=self.estimator)
        return covmats

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)

###############################################################################


class CospCovariances(BaseEstimator, TransformerMixin):

    """
    compute the cospectral covariance matrices

    """

    def __init__(
            self,
            window=128,
            overlap=0.75,
            fmin=None,
            fmax=None,
            fs=None,
            phase_correction=False):
        self._window = _nextpow2(window)
        self._overlap = overlap
        self._fmin = fmin
        self._fmax = fmax
        self._fs = fs

        self._phase_corr = phase_correction

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        Nt, Ne, _ = X.shape
        out = []

        for i in range(Nt):
            S = cospectrum(X[i], window=self._window, overlap=self._overlap,
                           fmin=self._fmin, fmax=self._fmax, fs=self._fs)
            out.append(S.real)

        return numpy.array(out)

    def fit_transform(self, X, y=None):
        return self.transform(X)


##########################################################################
def _nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n
