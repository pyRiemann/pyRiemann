"""Spatial filtering function."""
import numpy as np
from scipy.linalg import eigh, inv
from sklearn.base import BaseEstimator, TransformerMixin

from .utils.covariance import _check_est, normalize, get_nondiag_weight
from .utils.mean import mean_covariance
from .utils.ajd import ajd_pham
from .utils.mean import _check_mean_method
from . import estimation as est
from .preprocessing import Whitening


class Xdawn(BaseEstimator, TransformerMixin):
    """Xdawn algorithm.

    Xdawn is a spatial filtering method designed to improve the signal
    to signal + noise ratio (SSNR) of the ERP responses. Xdawn was originaly
    designed for P300 evoked potential by enhancing the target response with
    respect to the non-target response. This implementation is a generalization
    to any type of ERP.

    Parameters
    ----------
    nfilter : int, default=4
        The number of components to decompose M/EEG signals.
    classes : list of int | None, default=None
        List of classes to take into account for Xdawn.
        If None, all classes will be accounted.
    estimator : string, default='scm'
        Covariance matrix estimator, see
        :func:`pyriemann.utils.covariance.covariances`.
    baseline_cov : None | array, shape(n_chan, n_chan), default=None
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
    .. [1] Rivet, B., Souloumiac, A., Attina, V., & Gibert, G. (2009). xDAWN
        algorithm to enhance evoked potentials: application to brain-computer
        interface. Biomedical Engineering, IEEE Transactions on, 56(8),
        2035-2043.

    .. [2] Rivet, B., Cecotti, H., Souloumiac, A., Maby, E., & Mattout, J.
        (2011). Theoretical analysis of xDAWN algorithm: application to an
        efficient sensor selection in a P300 BCI. In Signal Processing
        Conference, 2011 19th European (pp. 1382-1386). IEEE.
    """

    def __init__(self, nfilter=4, classes=None, estimator='scm',
                 baseline_cov=None):
        """Init."""
        self.nfilter = nfilter
        self.classes = classes
        self.estimator = estimator
        self.baseline_cov = baseline_cov

    @property
    def estimator_fn(self):
        return _check_est(self.estimator)

    def fit(self, X, y):
        """Train Xdawn spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Set of trials.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : Xdawn instance
            The Xdawn instance.
        """
        n_trials, n_channels, n_times = X.shape

        self.classes_ = (np.unique(y) if self.classes is None else
                         self.classes)

        Cx = self.baseline_cov
        if Cx is None:
            # FIXME : too many reshape operation
            tmp = X.transpose((1, 2, 0))
            Cx = np.asarray(self.estimator_fn(
                tmp.reshape(n_channels, n_times * n_trials)
            ))

        self.evokeds_ = []
        self.filters_ = []
        self.patterns_ = []
        for c in self.classes_:
            # Prototyped response for each class
            P = np.mean(X[y == c, :, :], axis=0)

            # Covariance matrix of the prototyper response & signal
            C = np.asarray(self.estimator_fn(P))

            # Spatial filters
            evals, evecs = eigh(C, Cx)
            evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors
            evecs /= np.apply_along_axis(np.linalg.norm, 0, evecs)
            V = evecs
            A = np.linalg.pinv(V.T)
            # create the reduced prototyped response
            self.filters_.append(V[:, 0:self.nfilter].T)
            self.patterns_.append(A[:, 0:self.nfilter].T)
            self.evokeds_.append(np.dot(V[:, 0:self.nfilter].T, P))

        self.evokeds_ = np.concatenate(self.evokeds_, axis=0)
        self.filters_ = np.concatenate(self.filters_, axis=0)
        self.patterns_ = np.concatenate(self.patterns_, axis=0)
        return self

    def transform(self, X):
        """Apply spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_times)
            Set of trials.

        Returns
        -------
        Xf : ndarray, shape (n_trials, n_filters * n_classes, n_times)
            Set of spatialy filtered trials.
        """
        X = np.dot(self.filters_, X)
        X = X.transpose((1, 0, 2))
        return X


class BilinearFilter(BaseEstimator, TransformerMixin):
    r"""Bilinear spatial filter.

    Bilinear spatial filter for SPD matrices allows to define a custom spatial
    filter for bilinear projection of the data:

    .. math::
        \mathbf{Cf}_i = \mathbf{V} \mathbf{C}_i \mathbf{V}^T

    If log parameter is set to true, will return the log of the diagonal:

    .. math::
        \mathbf{cf}_i = \log [ \mathrm{diag} (\mathbf{Cf}_i) ]

    Parameters
    ----------
    filters : ndarray, shape (n_filters x n_channels)
        The filters for bilinear transform.
    log : bool, default=False
        If true, return the log variance, otherwise return the spatially
        filtered covariance matrices.

    Attributes
    ----------
    filters_ : ndarray
        If fit, the filter components used to decompose the data for each event
        type, concatenated, else empty.
    """

    def __init__(self, filters, log=False):
        """Init."""
        self.filters_ = filters
        self.filters = filters
        self.log = log

    def fit(self, X, y):
        """Train BilinearFilter spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : BilinearFilter instance
            The BilinearFilter instance.
        """
        if not isinstance(self.filters, np.ndarray):
            raise TypeError('filters must be an array.')
        if not isinstance(self.log, bool):
            raise TypeError('log must be a boolean')
        self.filters_ = self.filters
        return self

    def transform(self, X):
        """Apply spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.

        Returns
        -------
        Xf : ndarray, shape (n_trials, n_filters)
            Set of spatialy filtered log-variance or covariance depending
            on the 'log' input parameter.
        """
        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        if X[0].shape[1] != self.filters_.shape[1]:
            raise ValueError("Data and filters dimension must be compatible.")

        X_filt = np.dot(np.dot(self.filters_, X), self.filters_.T)
        X_filt = X_filt.transpose((1, 0, 2))

        # if logvariance
        if self.log:
            out = np.zeros((len(X_filt), len(self.filters_)))
            for i, x in enumerate(X_filt):
                out[i] = np.log(np.diag(x))
            return out
        else:
            return X_filt


class CSP(BilinearFilter):
    """CSP spatial filtering with covariance matrices as inputs.

    Implementation of the famous Common Spatial Pattern algorithm [1]_ [2]_,
    but with covariance matrices as input. In addition, the implementation
    allows different metric for the estimation of the class-related mean
    covariance matrices, as described in [3]_.

    This implementation support multiclass CSP by means of approximate joint
    diagonalization. In this case, the spatial filter selection is achieved
    according to [4]_.

    Parameters
    ----------
    nfilter : int, default=4
        The number of components to decompose M/EEG signals.
    metric : str, default='euclid'
        The metric for the estimation of mean covariance matrices.
    log : bool, default=True
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
    .. [1] Zoltan J. Koles, Michael S. Lazar, Steven Z. Zhou. Spatial Patterns
        Underlying Population Differences in the Background EEG. Brain
        Topography 2(4), 275-284, 1990.

    .. [2] Benjamin Blankertz, Ryota Tomioka, Steven Lemm, Motoaki Kawanabe,
        Klaus-Robert Muller. Optimizing Spatial Filters for Robust EEG
        Single-Trial Analysis. IEEE Signal Processing Magazine 25(1), 41-56,
        2008.

    .. [3] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, Common Spatial
        Pattern revisited by Riemannian geometry, IEEE International Workshop
        on Multimedia Signal Processing (MMSP), p. 472-476, 2010.

    .. [4] Grosse-Wentrup, Moritz, and Martin Buss. "Multiclass common spatial
        patterns and information theoretic feature extraction." Biomedical
        Engineering, IEEE Transactions on 55, no. 8 (2008): 1991-2000.
    """

    def __init__(self, nfilter=4, metric='euclid', log=True):
        """Init."""
        self.nfilter = nfilter
        self.metric = metric
        self.log = log

    def fit(self, X, y):
        """Train CSP spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Labels for each trial.

        Returns
        -------
        self : CSP instance
            The CSP instance.
        """
        if not isinstance(self.nfilter, int):
            raise TypeError('nfilter must be an integer')
        _check_mean_method(self.metric)
        if not isinstance(self.log, bool):
            raise TypeError('log must be a boolean')

        if not isinstance(X, (np.ndarray, list)):
            raise TypeError('X must be an array.')
        if not isinstance(y, (np.ndarray, list)):
            raise TypeError('y must be an array.')
        X, y = np.asarray(X), np.asarray(y)
        if X.ndim != 3:
            raise ValueError('X must be n_trials * n_channels * n_channels')
        if len(y) != len(X):
            raise ValueError('X and y must have the same length.')
        if np.squeeze(y).ndim != 1:
            raise ValueError('y must be of shape (n_trials,).')

        n_trials, n_channels, _ = X.shape
        classes = np.unique(y)
        # estimate class means
        C = []
        for c in classes:
            C.append(mean_covariance(X[y == c], self.metric))
        C = np.array(C)

        # Switch between binary and multiclass
        if len(classes) == 2:
            evals, evecs = eigh(C[1], C[0] + C[1])
            # sort eigenvectors
            ix = np.argsort(np.abs(evals - 0.5))[::-1]
        elif len(classes) > 2:
            evecs, D = ajd_pham(C)
            Ctot = np.array(mean_covariance(C, self.metric))
            evecs = evecs.T

            # normalize
            for i in range(evecs.shape[1]):
                tmp = np.dot(np.dot(evecs[:, i].T, Ctot), evecs[:, i])
                evecs[:, i] /= np.sqrt(tmp)

            mutual_info = []
            # class probability
            Pc = [np.mean(y == c) for c in classes]
            for j in range(evecs.shape[1]):
                a = 0
                b = 0
                for i, c in enumerate(classes):
                    tmp = np.dot(np.dot(evecs[:, j].T, C[i]),
                                 evecs[:, j])
                    a += Pc[i] * np.log(np.sqrt(tmp))
                    b += Pc[i] * (tmp ** 2 - 1)
                mi = - (a + (3.0 / 16) * (b ** 2))
                mutual_info.append(mi)
            ix = np.argsort(mutual_info)[::-1]
        else:
            raise ValueError("Number of classes must be >= 2.")

        # sort eigenvectors
        evecs = evecs[:, ix]

        # spatial patterns
        A = np.linalg.pinv(evecs.T)

        self.filters_ = evecs[:, 0:self.nfilter].T
        self.patterns_ = A[:, 0:self.nfilter].T

        return self


class SPoC(CSP):
    """SPoC spatial filtering with covariance matrices as inputs.

    Source Power Comodulation (SPoC) [1]_ allows to extract spatial filters and
    patterns by using a target (continuous) variable in the decomposition
    process in order to give preference to components whose power comodulates
    with the target variable.

    SPoC can be seen as an extension of the
    :class:`pyriemann.spatialfilters.CSP` driven by a continuous
    variable rather than a discrete (often binary) variable. Typical
    applications include extraction of motor patterns using EMG power or audio
    paterns using sound envelope.

    Parameters
    ----------
    nfilter : int, default=4
        The number of components to decompose M/EEG signals.
    metric : str, default='euclid'
        The metric for the estimation of mean covariance matrices.
    log : bool, default=True
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
    .. [1] Dahne, S., Meinecke, F. C., Haufe, S., Hohne, J., Tangermann, M.,
        Muller, K. R., & Nikulin, V. V. (2014). SPoC: a novel framework for
        relating the amplitude of neuronal oscillations to behaviorally
        relevant parameters. NeuroImage, 86, 111-122.
    """

    def fit(self, X, y):
        """Train spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels)
            Set of covariance matrices.
        y : ndarray, shape (n_trials,)
            Target variable for each trial.

        Returns
        -------
        self : SPoC instance
            The SPoC instance.
        """

        # Normalize target variable
        target = np.float64(y.copy())
        target -= target.mean()
        target /= target.std()

        C = mean_covariance(X, self.metric)
        Ce = np.zeros_like(X)
        for i in range(Ce.shape[0]):
            Ce[i] = X[i] * target[i]
        Cz = mean_covariance(Ce, self.metric)

        # solve eigenvalue decomposition
        evals, evecs = eigh(Cz, C)
        evals = evals.real
        evecs = evecs.real
        # sort vectors
        ix = np.argsort(np.abs(evals))[::-1]

        # sort eigenvectors
        evecs = evecs[:, ix]

        # spatial patterns
        A = np.linalg.pinv(evecs.T)

        self.filters_ = evecs[:, 0:self.nfilter].T
        self.patterns_ = A[:, 0:self.nfilter].T

        return self


class AJDC(BaseEstimator, TransformerMixin):
    """AJDC algorithm.

    The approximate joint diagonalization of Fourier cospectral matrices (AJDC)
    [1]_ is a versatile tool for blind source separation (BSS) tasks based on
    Second-Order Statistics (SOS), estimating spectrally uncorrelated sources.

    It can be applied:

    * on a single subject, to solve the classical BSS problem [1]_,
    * on several subjects, to solve the group BSS (gBSS) problem [2]_,
    * on several experimental conditions (for eg, baseline versus task), to
      exploit the diversity of source energy between conditions in addition
      to generic coloration and time-varying energy [1]_.

    AJDC estimates Fourier cospectral matrices by the Welch's method, and
    applies a trace-normalization. If necessary, it averages cospectra across
    subjects, and concatenates them along experimental conditions.
    Then, a dimension reduction and a whitening are applied on cospectra.
    An approximate joint diagonalization (AJD) [3]_ allows to estimate the
    joint diagonalizer, not constrained to be orthogonal. Finally, forward and
    backward spatial filters are computed.

    Parameters
    ----------
    window : int, default=128
        The length of the FFT window used for spectral estimation.
    overlap : float, default=0.5
        The percentage of overlap between window.
    fmin : float | None, default=None
        The minimal frequency to be returned. Since BSS models assume zero-mean
        processes, the first cospectrum (0 Hz) must be excluded.
    fmax : float | None, default=None
        The maximal frequency to be returned.
    fs : float | None, default=None
        The sampling frequency of the signal.
    dim_red : None | dict, default=None
        Parameter for dimension reduction of cospectra, because Pham's AJD is
        sensitive to matrices conditioning. For more details, see parameter
        ``dim_red`` of :class:`pyriemann.preprocessing.Whitening`.
    verbose : bool, default=True
        Verbose flag.

    Attributes
    ----------
    n_channels_ : int
        If fit, the number of channels of the signal.
    freqs_ : ndarray, shape (n_freqs,)
        If fit, the frequencies associated to cospectra.
    n_sources_ : int
        If fit, the number of components of the source space.
    forward_filters_ : ndarray, shape ``(n_sources_, n_channels_)``
        If fit, the spatial filters used to transform signal into source,
        also called deximing or separating matrix.
    backward_filters_ : ndarray, shape ``(n_channels_, n_sources_)``
        If fit, the spatial filters used to transform source into signal,
        also called mixing matrix.

    Notes
    -----
    .. versionadded:: 0.2.7

    See Also
    --------
    CospCovariances

    References
    ----------
    .. [1] M. Congedo, C. Gouy-Pailler, C. Jutten, "On the blind source
        separation of human electroencephalogram by approximate joint
        diagonalization of second order statistics", Clin Neurophysiol, 2008

    .. [2] M. Congedo, R. John, D. De Ridder, L. Prichep, "Group indepedent
        component analysis of resting state EEG in large normative samples",
        Int J Psychophysiol, 2010

    .. [3] D.-T. Pham, "Joint approximate diagonalization of positive definite
        Hermitian matrices", SIAM J Matrix Anal Appl, 2001

    """

    def __init__(self, window=128, overlap=0.5, fmin=None, fmax=None, fs=None,
                 dim_red=None, verbose=True):
        """Init."""

        self.window = window
        self.overlap = overlap
        self.fmin = fmin
        self.fmax = fmax
        self.fs = fs
        self.dim_red = dim_red
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit.

        Compute and diagonalize cospectra, to estimate forward and backward
        spatial filters.

        Parameters
        ----------
        X : ndarray, shape (n_subjects, n_conditions, n_channels, n_times) | \
                list of n_subjects of list of n_conditions ndarray of shape \
                (n_channels, n_times), with same n_conditions and n_channels \
                but different n_times
            Multi-channel time-series in channel space, acquired for different
            subjects and under different experimental conditions.
        y : None
            Currently not used, here for compatibility with sklearn API.

        Returns
        -------
        self : AJDC instance
            The AJDC instance.
        """
        # definition of params for Welch's method
        cospcov = est.CospCovariances(
            window=self.window,
            overlap=self.overlap,
            fmin=self.fmin,
            fmax=self.fmax,
            fs=self.fs)
        # estimation of cospectra on subjects and conditions
        cosp = []
        for s in range(len(X)):
            cosp_ = cospcov.transform(X[s])
            if s == 0:
                n_conditions = cosp_.shape[0]
                self.n_channels_ = cosp_.shape[1]
                self.freqs_ = cospcov.freqs_
            else:
                if n_conditions != cosp_.shape[0]:
                    raise ValueError('Unequal number of conditions')
                if self.n_channels_ != cosp_.shape[1]:
                    raise ValueError('Unequal number of channels')
            cosp.append(cosp_)
        cosp = np.transpose(np.array(cosp), axes=(0, 1, 4, 2, 3))

        # trace-normalization of cospectra, Eq(3) in [2]
        cosp = normalize(cosp, "trace")
        # average of cospectra across subjects, Eq(7) in [2]
        cosp = np.mean(cosp, axis=0, keepdims=False)
        # concatenation of cospectra along conditions
        self._cosp_channels = np.concatenate(cosp, axis=0)
        # estimation of non-diagonality weights, Eq(B.1) in [1]
        weights = get_nondiag_weight(self._cosp_channels)

        # dimension reduction and whitening, Eq.(8) in [2], computed on the
        # weighted mean of cospectra across frequencies (and conditions)
        whit = Whitening(
            metric='euclid',
            dim_red=self.dim_red,
            verbose=self.verbose)
        cosp_rw = whit.fit_transform(self._cosp_channels, weights)
        self.n_sources_ = whit.n_components_

        # approximate joint diagonalization, currently by Pham's algorithm [3]
        diag_filters, self._cosp_sources = ajd_pham(
            cosp_rw,
            n_iter_max=100,
            sample_weight=weights)

        # computation of forward and backward filters, Eq.(9) and (10) in [2]
        self.forward_filters_ = diag_filters @ whit.filters_.T
        self.backward_filters_ = whit.inv_filters_.T @ inv(diag_filters)
        return self

    def transform(self, X):
        """Transform channel space to source space.

        Transform channel space to source space, applying forward spatial
        filters.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series in channel space.

        Returns
        -------
        source : ndarray, shape (n_matrices, n_sources, n_times)
            Multi-channel time-series in source space.
        """
        if X.ndim != 3:
            raise ValueError('X must have 3 dimensions (Got %d)' % X.ndim)
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                'X does not have the good number of channels. Should be %d but'
                ' got %d.' % (self.n_channels_, X.shape[1]))

        source = self.forward_filters_ @ X
        return source

    def inverse_transform(self, X, supp=None):
        """Transform source space to channel space.

        Transform source space to channel space, applying backward spatial
        filters, with the possibility to suppress some sources, like in BSS
        filtering/denoising.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_sources, n_times)
            Multi-channel time-series in source space.
        supp : list of int | None, default=None
            Indices of sources to suppress. If None, no source suppression.

        Returns
        -------
        signal : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series in channel space.
        """
        if X.ndim != 3:
            raise ValueError('X must have 3 dimensions (Got %d)' % X.ndim)
        if X.shape[1] != self.n_sources_:
            raise ValueError(
                'X does not have the good number of sources. Should be %d but '
                'got %d.' % (self.n_sources_, X.shape[1]))

        denois = np.eye(self.n_sources_)
        if supp is None:
            pass
        elif isinstance(supp, list):
            for s in supp:
                denois[s, s] = 0
        else:
            raise ValueError('Parameter supp must be a list of int, or None')

        signal = self.backward_filters_ @ denois @ X
        return signal

    def get_src_expl_var(self, X):
        """Estimate explained variances of sources.

        Estimate explained variances of sources, see Appendix D in [1].

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_times)
            Multi-channel time-series in channel space.

        Returns
        -------
        src_var : ndarray, shape (n_matrices, n_sources)
            Explained variance for each source.
        """
        if X.ndim != 3:
            raise ValueError('X must have 3 dimensions (Got %d)' % X.ndim)
        if X.shape[1] != self.n_channels_:
            raise ValueError(
                'X does not have the good number of channels. Should be %d but'
                ' got %d.' % (self.n_channels_, X.shape[1]))

        cov = est.Covariances().transform(X)

        src_var = np.zeros((X.shape[0], self.n_sources_))
        for s in range(self.n_sources_):
            src_var[:, s] = np.trace(
                self.backward_filters_[:, s] * self.forward_filters_[s].T * cov
                * self.forward_filters_[s] * self.backward_filters_[:, s].T,
                axis1=-2,
                axis2=-1)
        return src_var
