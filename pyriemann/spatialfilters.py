"""Spatial filtering function."""
import numpy

from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from .utils.covariance import _check_est
from .utils.mean import mean_covariance
from .utils.ajd import ajd_pham
from .utils.mean import _check_mean_method


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
    baseline_cov : array, shape(n_chan, n_chan) | None (default)
        Covariance matrix to which the average signals are compared. If None,
        the baseline covariance is computed across all trials and time samples.
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

    def __init__(self, nfilter=4, classes=None, estimator='scm',
                 baseline_cov=None):
        """Init."""
        self.nfilter = nfilter
        self.classes = classes
        self.estimator = _check_est(estimator)
        self.baseline_cov = baseline_cov

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

        self.classes_ = (numpy.unique(y) if self.classes is None else
                         self.classes)

        Cx = self.baseline_cov
        if Cx is None:
            # FIXME : too many reshape operation
            tmp = X.transpose((1, 2, 0))
            Cx = numpy.matrix(self.estimator(tmp.reshape(Ne, Ns * Nt)))

        self.evokeds_ = []
        self.filters_ = []
        self.patterns_ = []
        for c in self.classes_:
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


class BilinearFilter(BaseEstimator, TransformerMixin):
    """ Bilinear spatial filter.

    Bilinear spatial filter for covariance matrices.
    allow to define a custom spatial filter for bilinear projection of the
    data :

    .. math::
        \mathbf{Cf}_i = \mathbf{V} \mathbf{C}_i \mathbf{V}^T

    if log parameter is set to true, will return the log of the diagonal :

    .. math::
        \mathbf{cf}_i = \log \left[ \diag (\mathbf{Cf}_i) \\right]

    Parameters
    ----------
    filters: ndarray of shape (Nfilters x Nchannels)
        the filters for bilinear transform
    log : bool (default False)
        If true, return the log variance, otherwise return the spatially
        filtered covariance matrices.

    Attributes
    ----------
    filters_ : ndarray
        If fit, the Xdawn components used to decompose the data for each event
        type, concatenated, else empty.
    """

    def __init__(self, filters, log=False):
        """Init."""
        if not isinstance(filters, numpy.ndarray):
            raise TypeError('filters must be an array.')
        self.filters_ = filters
        self.filters = filters
        if not isinstance(log, bool):
            raise TypeError('log must be a boolean')
        self.log = log

    def fit(self, X, y):
        """Train CSP spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : CSP instance
            The CSP instance.
        """
        self.filters_ = self.filters
        return self

    def transform(self, X):
        """Apply spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance.

        Returns
        -------
        Xf : ndarray, shape (n_trials, n_filters)
             ndarray of spatialy filtered log-variance or covariance depending
             on the 'log' input paramter.
        """
        if not isinstance(X, (numpy.ndarray, list)):
            raise TypeError('X must be an array.')
        if X[0].shape[1] != self.filters_.shape[1]:
            raise ValueError("Data and filters dimension must be compatible.")

        X_filt = numpy.dot(numpy.dot(self.filters_, X), self.filters_.T)
        X_filt = X_filt.transpose((1, 0, 2))

        # if logvariance
        if self.log:
            out = numpy.zeros((len(X_filt), len(self.filters_)))
            for i, x in enumerate(X_filt):
                out[i] = numpy.log(numpy.diag(x))
            return out
        else:
            return X_filt


class CSP(BilinearFilter):
    """Implementation of the CSP spatial Filtering with Covariance as input.

    Implementation of the famous Common Spatial Pattern Algorithm, but with
    covariance matrices as input. In addition, the implementation allow
    different metric for the estimation of the class-related mean covariance
    matrices, as described in [3].

    This implementation support multiclass CSP by means of approximate joint
    diagonalization. In this case, the spatial filter selection is achieved
    according to [4].

    Parameters
    ----------
    nfilter : int (default 4)
        The number of components to decompose M/EEG signals.
    metric : str (default "euclid")
        The metric for the estimation of mean covariance matrices
    log : bool (default True)
        If true, return the log variance, otherwise return the spatially
        filtered covariance matrices.

    Attributes
    ----------
    filters_ : ndarray
        If fit, the CSP spatial filters, else None.
    patterns_ : ndarray
        If fit, the CSP spatial patterns, else None.


    See Also
    --------
    MDM, SPoC

    References
    ----------
    [1] Zoltan J. Koles, Michael S. Lazar, Steven Z. Zhou. Spatial Patterns
        Underlying Population Differences in the Background EEG. Brain
        Topography 2(4), 275-284, 1990.
    [2] Benjamin Blankertz, Ryota Tomioka, Steven Lemm, Motoaki Kawanabe,
        Klaus-Robert Muller. Optimizing Spatial Filters for Robust EEG
        Single-Trial Analysis. IEEE Signal Processing Magazine 25(1), 41-56,
        2008.
    [3] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, Common Spatial
        Pattern revisited by Riemannian geometry, IEEE International Workshop
        on Multimedia Signal Processing (MMSP), p. 472-476, 2010.
    [4] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common spatial
        patterns and information theoretic feature extraction." Biomedical
        Engineering, IEEE Transactions on 55, no. 8 (2008): 1991-2000.
    """

    def __init__(self, nfilter=4, metric='euclid', log=True):
        """Init."""
        if not isinstance(nfilter, int):
            raise TypeError('nfilter must be an integer')
        self.nfilter = nfilter
        _check_mean_method(metric)
        self.metric = metric
        if not isinstance(log, bool):
            raise TypeError('log must be a boolean')
        self.log = log

    def fit(self, X, y):
        """Train CSP spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.

        Returns
        -------
        self : CSP instance
            The CSP instance.
        """
        if not isinstance(X, (numpy.ndarray, list)):
            raise TypeError('X must be an array.')
        if not isinstance(y, (numpy.ndarray, list)):
            raise TypeError('y must be an array.')
        X, y = numpy.asarray(X), numpy.asarray(y)
        if X.ndim != 3:
            raise ValueError('X must be n_trials * n_channels * n_channels')
        if len(y) != len(X):
            raise ValueError('X and y must have the same length.')
        if numpy.squeeze(y).ndim != 1:
            raise ValueError('y must be of shape (n_trials,).')

        Nt, Ne, Ns = X.shape
        classes = numpy.unique(y)
        # estimate class means
        C = []
        for c in classes:
            C.append(mean_covariance(X[y == c], self.metric))
        C = numpy.array(C)

        # Switch between binary and multiclass
        if len(classes) == 2:
            evals, evecs = eigh(C[1], C[0] + C[1])
            # sort eigenvectors
            ix = numpy.argsort(numpy.abs(evals - 0.5))[::-1]
        elif len(classes) > 2:
            evecs, D = ajd_pham(C)
            Ctot = numpy.array(mean_covariance(C, self.metric))
            evecs = evecs.T

            # normalize
            for i in range(evecs.shape[1]):
                tmp = numpy.dot(numpy.dot(evecs[:, i].T, Ctot), evecs[:, i])
                evecs[:, i] /= numpy.sqrt(tmp)

            mutual_info = []
            # class probability
            Pc = [numpy.mean(y == c) for c in classes]
            for j in range(evecs.shape[1]):
                a = 0
                b = 0
                for i, c in enumerate(classes):
                    tmp = numpy.dot(numpy.dot(evecs[:, j].T, C[i]),
                                    evecs[:, j])
                    a += Pc[i] * numpy.log(numpy.sqrt(tmp))
                    b += Pc[i] * (tmp ** 2 - 1)
                mi = - (a + (3.0 / 16) * (b ** 2))
                mutual_info.append(mi)
            ix = numpy.argsort(mutual_info)[::-1]
        else:
            raise ValueError("Number of classes must be >= 2.")

        # sort eigenvectors
        evecs = evecs[:, ix]

        # spatial patterns
        A = numpy.linalg.pinv(evecs.T)

        self.filters_ = evecs[:, 0:self.nfilter].T
        self.patterns_ = A[:, 0:self.nfilter].T

        return self


class SPoC(CSP):
    """Implementation of the SPoC spatial filtering with Covariance as input.

    Source Power Comodulation (SPoC) [1] allow to extract spatial filters and
    patterns by using a target (continuous) variable in the decomposition
    process in order to give preference to components whose power comodulates
    with the target variable.

    SPoC can be seen as an extension of the `CSP` driven by a continuous
    variable rather than a discrete (often binary) variable. Typical
    applications include extraction of motor patterns using EMG power or audio
    paterns using sound envelope.

    Parameters
    ----------
    nfilter : int (default 4)
        The number of components to decompose M/EEG signals.
    metric : str (default "euclid")
        The metric for the estimation of mean covariance matrices
    log : bool (default True)
        If true, return the log variance, otherwise return the spatially
        filtered covariance matrices.

    Attributes
    ----------
    filters_ : ndarray
        If fit, the SPoC spatial filters, else None.
    patterns_ : ndarray
        If fit, the SPoC spatial patterns, else None.

    Notes
    -----
    .. versionadded:: 0.2.4

    See Also
    --------
    CSP, SPoC

    References
    ----------
    [1] Dahne, S., Meinecke, F. C., Haufe, S., Hohne, J., Tangermann, M.,
        Muller, K. R., & Nikulin, V. V. (2014). SPoC: a novel framework for
        relating the amplitude of neuronal oscillations to behaviorally
        relevant parameters. NeuroImage, 86, 111-122.
    """

    def fit(self, X, y):
        """Train spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of covariance.
        y : ndarray shape (n_trials, 1)
            target variable corresponding to each trial.

        Returns
        -------
        self : SPoC instance
            The SPoC instance.
        """

        # Normalize target variable
        target = numpy.float64(y.copy())
        target -= target.mean()
        target /= target.std()

        C = mean_covariance(X, self.metric)
        Ce = numpy.zeros_like(X)
        for i in range(Ce.shape[0]):
            Ce[i] = X[i] * target[i]
        Cz = mean_covariance(Ce, self.metric)

        # solve eigenvalue decomposition
        evals, evecs = eigh(Cz, C)
        evals = evals.real
        evecs = evecs.real
        # sort vectors
        ix = numpy.argsort(numpy.abs(evals))[::-1]

        # sort eigenvectors
        evecs = evecs[:, ix]

        # spatial patterns
        A = numpy.linalg.pinv(evecs.T)

        self.filters_ = evecs[:, 0:self.nfilter].T
        self.patterns_ = A[:, 0:self.nfilter].T

        return self
