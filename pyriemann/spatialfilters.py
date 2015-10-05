"""Spatial filtering function."""
import numpy

from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from .utils.covariance import _check_est


class Xdawn(BaseEstimator, TransformerMixin):

    """Implementation of the Xdawn Algorithm.

    Xdawn is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the ERP responses. Xdawn was originaly
    designed for P300 evoked potential by enhancing the target response with
    respect to the non-target response. This implementation is a generalization
    to any type of ERP.

    Parameters
    ----------
    nfilter : int (default 4)
        The number of components to decompose M/EEG signals.
    classes : list of int | None (default None)
        list of classes to take into account for xdawn. If None (default), all
        classes will be accounted.
    estimator : str (default 'scm')
        covariance matrix estimator. For regularization consider 'lwf' or 'oas'

    Attributes
    ----------
    filters_ : ndarray
        If fit, the Xdawn components used to decompose the data for each event
        type, concatenated, else empty.
    patterns_ : ndarray
        If fit, the Xdawn patterns used to restore M/EEG signals for each event
        type, concatenated, else empty.
    evokeds_ : ndarray
        If fit, the evoked response for each event type, concatenated.


    See Also
    --------
    XdawnCovariances

    References
    ----------
    [1] Rivet, B., Souloumiac, A., Attina, V., & Gibert, G. (2009). xDAWN
    algorithm to enhance evoked potentials: application to brain-computer
    interface. Biomedical Engineering, IEEE Transactions on, 56(8), 2035-2043.
    [2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., & Mattout, J. (2011,
    August). Theoretical analysis of xDAWN algorithm: application to an
    efficient sensor selection in a P300 BCI. In Signal Processing Conference,
    2011 19th European (pp. 1382-1386). IEEE.
    """

    def __init__(self, nfilter=4, classes=None, estimator='scm'):
        """Init."""
        self.nfilter = nfilter
        self.classes = classes
        self.estimator = _check_est(estimator)

    def fit(self, X, y):
        """Train xdawn spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : Xdawn instance
            The Xdawn instance.
        """
        Nt, Ne, Ns = X.shape

        if self.classes is None:
            self.classes = numpy.unique(y)

        # FIXME : too many reshape operation
        tmp = X.transpose((1, 2, 0))
        Cx = numpy.matrix(self.estimator(tmp.reshape(Ne, Ns * Nt)))

        self.evokeds_ = []
        self.filters_ = []
        self.patterns_ = []
        for c in self.classes:
            # Prototyped responce for each class
            P = numpy.mean(X[y == c, :, :], axis=0)

            # Covariance matrix of the prototyper response & signal
            C = numpy.matrix(self.estimator(P))

            # Spatial filters
            evals, evecs = eigh(C, Cx)
            evecs = evecs[:, numpy.argsort(evals)[::-1]]  # sort eigenvectors
            evecs /= numpy.apply_along_axis(numpy.linalg.norm, 0, evecs)
            V = evecs
            A = numpy.linalg.pinv(V.T)
            # create the reduced prototyped response
            self.filters_.append(V[:, 0:self.nfilter].T)
            self.patterns_.append(A[:, 0:self.nfilter].T)
            self.evokeds_.append(numpy.dot(V[:, 0:self.nfilter].T, P))

        self.evokeds_ = numpy.concatenate(self.evokeds_, axis=0)
        self.filters_ = numpy.concatenate(self.filters_, axis=0)
        self.patterns_ = numpy.concatenate(self.patterns_, axis=0)
        return self

    def transform(self, X):
        """Apply spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            ndarray of trials.

        Returns
        -------
        Xf : ndarray, shape (n_trials, n_filters * n_classes, n_samples)
            ndarray of spatialy filtered trials.
        """
        X = numpy.dot(self.filters_, X)
        X = X.transpose((1, 0, 2))
        return X
