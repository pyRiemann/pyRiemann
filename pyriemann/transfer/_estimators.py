import warnings

from joblib import Parallel, delayed
import numpy as np
from sklearn.base import (
    BaseEstimator,
    TransformerMixin,
    is_classifier,
    is_regressor
)
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, r2_score
from sklearn.pipeline import Pipeline

from ..optimization.grassmann import (
    _get_rotation_manifold, _get_rotation_tangentspace
)
from ._tools import decode_domains
from ..classification import MDM
from ..preprocessing import Whitening
from ..utils import deprecated
from ..utils.base import invsqrtm, powm, sqrtm
from ..utils.distance import distance
from ..utils.geodesic import geodesic
from ..utils.mean import mean_covariance, mean_riemann
from ..utils.utils import check_weights, check_metric


###############################################################################


def _check_inputs(X):
    if X.ndim not in [2, 3]:
        raise ValueError(f"Input must be a 2d or a 3d array (Got {X.ndim}).")


class TLDummy(TransformerMixin, BaseEstimator):
    """No transformation for transfer learning.

    No transformation of data between the domains.

    Notes
    -----
    .. versionadded:: 0.4
    """

    def fit(self, X, y_enc=None):
        """Do nothing.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : TLDummy instance
            The TLDummy instance.
        """
        return self

    def transform(self, X):
        """Do nothing.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Same data as in the input.
        """
        return X

    def fit_transform(self, X, y_enc=None):
        """Do nothing.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Same data as in the input.
        """
        return self.fit(X, y_enc).transform(X)


class TLCenter(TransformerMixin, BaseEstimator):
    """Centering for transfer learning.

    For inputs in matrix manifold, it recenters the matrices from each domain
    to the identity matrix on manifold, ie it makes the mean of the matrices of
    each domain become the identity [1]_.
    This operation corresponds to a whitening when the matrices represent the
    spatial covariance matrices of multivariate signals.

    For inputs in tangent space, it recenters the tangent vectors from each
    domain to the origin of tangent space, ie it makes the mean of the vectors
    of each domain become zero.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training set (target and source domains),
       and .transform() on the target domain of the test set.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target in ``transform()`` function:

        * if not empty, ``transform()`` recenters matrices to the specified
          target domain;
        * else, ``transform()`` recenters matrices to the last fitted domain.
    metric : str, default="riemann"
        For inputs in manifold,
        metric used for mean estimation. For the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`.
        Note, however, that only when using the "riemann" metric that we are
        ensured to re-center the matrices precisely to the identity.

    Attributes
    ----------
    centers_ : dict
        Dictionary with key=domain_name and value=domain_center.

    Notes
    -----
    .. versionadded:: 0.4
    .. versionchanged:: 0.8
        Add support for tangent space centering.

    References
    ----------
    .. [1] `Transfer Learning: A Riemannian Geometry Framework With
        Applications to Brain–Computer Interfaces
        <https://hal.archives-ouvertes.fr/hal-01923278/>`_
        P Zanini et al, IEEE Transactions on Biomedical Engineering, vol. 65,
        no. 5, pp. 1107-1116, August, 2017
    .. [2] `Transfer Learning for Brain-Computer Interfaces:
        A Euclidean Space Data Alignment Approach
        <https://arxiv.org/abs/1808.05464>`_
        He He and Dongrui Wu, IEEE Transactions on Biomedical Engineering, 2019
    """

    def __init__(self, target_domain, metric="riemann"):
        """Init"""
        self.target_domain = target_domain
        self.metric = metric

    @property
    @deprecated(
        "Attribute `recenter_` is deprecated and will be removed in 0.10.0; "
        "please use `centers_`."
    )
    def recenter_(self):
        return self.centers_

    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLCenter.

        For each domain, it calculates the mean of matrices or vectors of this
        domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.
        sample_weight : None | ndarray, shape (n_matrices,) or \
                shape (n_vectors,), default=None
            Weights for each matrix or vector. If None, it uses equal weights.

        Returns
        -------
        self : TLCenter instance
            The TLCenter instance.
        """
        _check_inputs(X)
        _, _, domains = decode_domains(X, y_enc)
        sample_weight = check_weights(sample_weight, X.shape[0])

        self.centers_ = {}

        for d in np.unique(domains):
            idx = domains == d

            if X.ndim == 3:
                self.centers_[d] = Whitening(metric=self.metric).fit(
                    X[idx],
                    sample_weight=sample_weight[idx],
                )

            else:
                self.centers_[d] = np.average(
                    X[idx],
                    axis=0,
                    weights=sample_weight[idx],
                )

        return self

    def transform(self, X):
        """Center in the target domain.

        .. note::
           This method is designed for using at test time,
           recentering all inputs in target domain, or in the last fitted
           domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_classes) or \
                shape (n_vectors, n_ts)
            Set of centered SPD matrices or tangent vectors in target domain.
        """
        _check_inputs(X)

        # if target domain is specified, use it
        if self.target_domain != "":
            target_domain = self.target_domain
        # else, use last calibrated domain as target domain
        else:
            target_domain = list(self.centers_.keys())[-1]

        if X.ndim == 3:
            X_new = self.centers_[target_domain].transform(X)
        else:
            X_new = X - self.centers_[self.target_domain]

        return X_new

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLCenter and then center each domain.

        For each domain, it calculates the mean of matrices or vectors of this
        domain, and then recenters them to identity matrix or to null vector.

        .. note::
           This method is designed for using at training time.
           The output for .fit_transform() will be different
           than using .fit() and .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.
        sample_weight : None | ndarray, shape (n_matrices,) or \
                shape (n_vectors,), default=None
            Weights for each matrix or vector. If None, it uses equal weights.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of centered SPD matrices or tangent vectors in each domain.
        """
        self.fit(X, y_enc, sample_weight=sample_weight)
        _, _, domains = decode_domains(X, y_enc)

        X_new = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d

            if X.ndim == 3:
                X_new[idx] = self.centers_[d].transform(X[idx])
            else:
                X_new[idx] = X[idx] - self.centers_[d]

        return X_new


class TLScale(TransformerMixin, BaseEstimator):
    """Scaling for transfer learning.

    For inputs in matrix manifold, it stretches the matrices from each domain
    around their mean so that the dispersion of the matrices of each domain is
    equal to one [1]_.

    For inputs in tangent space, it scales the tangent vectors from each domain
    so that the mean of norms of vectors of each domain is equal to one.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training set (target and source domains),
       and .transform() on the target domain of the test set.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    dispersion : float, default=1.0
        For inputs in manifold, target value for the dispersion of the
        matrices.
    centered_data : bool, default=False
        For inputs in manifold, whether the matrices have been re-centered to
        the identity matrix beforehand.
    metric : str, default="riemann"
        For inputs in manifold, metric used for calculating the dispersion.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.distance`.
        The stretching operation in manifold is properly defined only for the
        "riemann" metric.

    Attributes
    ----------
    scales_ : dict
        Dictionary with key=domain_name and value=domain_scale.

    See Also
    --------
    TLCenter

    Notes
    -----
    .. versionadded:: 0.4
    .. versionchanged:: 0.8
        Add support for tangent space scaling.

    References
    ----------
    .. [1] `Riemannian Procrustes analysis: transfer learning for
        brain-computer interfaces
        <https://hal.archives-ouvertes.fr/hal-01971856>`_
        PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering,
        vol. 66, no. 8, pp. 2390-2401, December, 2018
    """

    def __init__(
        self,
        target_domain,
        final_dispersion=1.0,
        centered_data=False,
        metric="riemann",
    ):
        """Init"""
        self.target_domain = target_domain
        self.final_dispersion = final_dispersion
        self.centered_data = centered_data
        self.metric = metric

    @property
    @deprecated(
        "Attribute `dispersions_` is deprecated and will be removed in 0.10.0;"
        " please use `scales_`."
    )
    def dispersions_(self):
        return self.scales_

    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLScale.

        For each domain, it calculates the scaling of this domain,
        ie the dispersion around the mean of matrices,
        or the mean of the norm of vectors.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.
        sample_weight : None | ndarray, shape (n_matrices,) or \
                shape (n_vectors,), default=None
            Weights for each matrix or vector. If None, it uses equal weights.

        Returns
        -------
        self : TLScale instance
            The TLScale instance.
        """
        _check_inputs(X)
        _, _, domains = decode_domains(X, y_enc)
        sample_weight = check_weights(sample_weight, X.shape[0])

        self._means, self.scales_ = {}, {}
        for d in np.unique(domains):
            idx = domains == d
            sample_weight_d = check_weights(sample_weight[idx], np.sum(idx))

            if X.ndim == 3:
                if self.centered_data:
                    self._means[d] = np.eye(X.shape[-1])
                else:
                    self._means[d] = mean_riemann(
                        X[idx], sample_weight=sample_weight_d
                    )
                dist = distance(
                    X[idx],
                    self._means[d],
                    metric=self.metric,
                    squared=True,
                )
                self.scales_[d] = np.sum(sample_weight_d * np.squeeze(dist))

            else:
                self.scales_[d] = np.average(
                    np.linalg.norm(X[idx], axis=1),
                    axis=0,
                    weights=sample_weight_d,
                )

        return self

    def _center(self, X, mean):
        Mean_isqrt = invsqrtm(mean)
        return Mean_isqrt @ X @ Mean_isqrt

    def _uncenter(self, X, mean):
        Mean_sqrt = sqrtm(mean)
        return Mean_sqrt @ X @ Mean_sqrt

    def _strech(self, X, dispersion_in, dispersion_out):
        return powm(X, np.sqrt(dispersion_out / dispersion_in))

    def transform(self, X):
        """Scale in the target domain.

        .. note::
           This method is designed for using at test time,
           scaling all inputs in target domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_classes) or \
                shape (n_vectors, n_ts)
            Set of scaled SPD matrices or tangent vectors in target domain.
        """
        _check_inputs(X)

        if X.ndim == 3:
            if not self.centered_data:  # center matrices to identity
                X = self._center(X, self._means[self.target_domain])

            # stretch
            X_new = self._strech(
                X, self.scales_[self.target_domain], self.final_dispersion
            )

            if not self.centered_data:  # re-center back to previous mean
                X_new = self._uncenter(X_new, self._means[self.target_domain])

        else:
            X_new = X / self.scales_[self.target_domain]

        return X_new

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLScale and then scale each domain.

        For each domain, it calculates the dispersion around the mean of this
        domain, and then stretches them to the desired final dispersion.
        For vectors, it scales them so that the mean of norms of vectors of
        each domain is equal to one.

        .. note::
           This method is designed for using at training time.
           The output for .fit_transform() will be different
           than using .fit() and .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.
        sample_weight : None | ndarray, shape (n_matrices,) or \
                shape (n_vectors,), default=None
            Weights for each matrix or vector. If None, it uses equal weights.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of scaled SPD matrices or tangent vectors in each domain.
        """
        self.fit(X, y_enc, sample_weight=sample_weight)
        _, _, domains = decode_domains(X, y_enc)

        X_new = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d

            if X.ndim == 3:
                if not self.centered_data:  # re-center matrices to identity
                    X[idx] = self._center(X[idx], self._means[d])

                # stretch
                X_new[idx] = self._strech(
                    X[idx], self.scales_[d], self.final_dispersion
                )

                if not self.centered_data:  # re-center back to previous mean
                    X_new[idx] = self._uncenter(X_new[idx], self._means[d])

            else:
                X_new[idx] = X[idx] / self.scales_[d]

        return X_new


@deprecated(
    "TLStretch is deprecated and will be removed in 0.10.0; "
    "please use TLScale."
)
class TLStretch(TLScale):
    pass


class TLRotate(TransformerMixin, BaseEstimator):
    """Rotation for transfer learning.

    For inputs in matrix manifold, it rotates the matrices from each source
    domain so to match its class means with those from the target domain.
    The loss function for this matching is described in [1]_ and the
    optimization procedure for minimizing it in [2]_.

    For inputs in tangent space, it rotates the tangent vectors from each
    source domain so to match its class means with those from the target domain
    [3]_.

    .. note::
       The inputs from each domain must have been centered to the before
       calculating the rotation.

    .. note::
       Using .fit() and then .transform() will give different results than
       .fit_transform(). In fact, .fit_transform() should be applied on the
       training set (target and source domains),
       and .transform() on the target domain of the test set.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    weights : None | ndarray, shape (n_classes,), default=None
        For inputs in manifold, weights to assign for each class.
        If None, it uses equal weights.
    metric : {"euclid", "riemann"}, default="euclid"
        For inputs in manifold, distance to minimize between class means.
    n_jobs : int, default=1
        For inputs in manifold, number of jobs to use for the computation.
        This works by computing the rotation matrix for each source domain in
        parallel. If -1 all CPUs are used.
    expl_var : float, default=0.999
        For inputs in tangent space, dimension reduction applied to the cross
        product matrix during Procrustes analysis.
        If float in (0,1], percentage of variance that needs to be explained.
        Else, number of components.
    n_components : int | "max", default=1
        For inputs in tangent space,
        parameter ``n_components`` used in ``sklearn.decomposition.PCA``.
        If int, number of components to keep in PCA.
        If "max", all components are kept.
    n_clusters : int, default=3
        For inputs in tangent space, number of clusters used to split data.

    Attributes
    ----------
    rotations_ : dict
        Dictionary with key=domain_name and value=domain_rotation_matrix.

    See Also
    --------
    TLCenter

    Notes
    -----
    .. versionadded:: 0.4
    .. versionchanged:: 0.8
        Added support for tangent space rotation.

    References
    ----------
    .. [1] `Riemannian Procrustes analysis: transfer learning for
        brain-computer interfaces
        <https://hal.archives-ouvertes.fr/hal-01971856>`_
        PLC Rodrigues et al, IEEE Transactions on Biomedical Engineering,
        vol. 66, no. 8, pp. 2390-2401, December, 2018
    .. [2] `An introduction to optimization on smooth manifolds
        <https://www.nicolasboumal.net/book/>`_
        N. Boumal. To appear with Cambridge University Press. June, 2022
    .. [3] `Tangent space alignment: Transfer learning for brain-computer
        interface
        <https://www.frontiersin.org/articles/10.3389/fnhum.2022.1049985/pdf>`_
        A. Bleuzé, J. Mattout and M. Congedo, Frontiers in Human Neuroscience,
        2022
    """

    def __init__(
        self,
        target_domain,
        weights=None,
        metric="euclid",
        n_jobs=1,
        expl_var=0.999,
        n_components=1,
        n_clusters=3,
    ):
        """Init"""
        self.target_domain = target_domain
        self.weights = weights
        self.metric = metric
        self.n_jobs = n_jobs
        self.expl_var = expl_var
        self.n_components = n_components
        self.n_clusters = n_clusters

    def fit(self, X, y_enc, sample_weight=None):
        """Fit TLRotate.

        It calculates the rotations matrices to transform each source domain
        into the target domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.
        sample_weight : None | ndarray, shape (n_matrices,) or \
                shape (n_vectors,), default=None
            Weights for each matrix or vector. If None, it uses equal weights.

        Returns
        -------
        self : TLRotate instance
            The TLRotate instance.
        """
        _check_inputs(X)
        _, _, domains = decode_domains(X, y_enc)
        sample_weight = check_weights(sample_weight, X.shape[0])

        if X.ndim == 3:
            self._fit_manifold(X, y_enc, domains, sample_weight)
        else:
            self._fit_tangentspace(X, y_enc, domains, sample_weight)

        return self

    def _fit_manifold(self, X, y_enc, domains, sample_weight):
        """Fit TLRotate on manifold.

        It computes the mean M of SPD matrices X for each class of each domain.
        It computes the rotation for each source domain so that its class means
        are close to those from the target domain.
        """
        idx = domains == self.target_domain
        X_target, y_target = X[idx], y_enc[idx]
        M_target = np.stack([
            mean_riemann(
                X_target[y_target == label],
                sample_weight=sample_weight[idx][y_target == label],
            ) for label in np.unique(y_target)
        ])

        source_domains = np.unique(domains)
        source_domains = source_domains[source_domains != self.target_domain]
        rotations = Parallel(n_jobs=self.n_jobs)(
            delayed(_get_rotation_manifold)(
                np.stack([
                    mean_riemann(
                        X[domains == d][y_enc[domains == d] == label],
                        sample_weight=sample_weight[domains == d][
                            y_enc[domains == d] == label
                        ]
                    ) for label in np.unique(y_enc[domains == d])
                ]),
                M_target,
                weights=self.weights,
                metric=self.metric,
            ) for d in source_domains
        )

        self.rotations_ = {}
        for d, rot in zip(source_domains, rotations):
            self.rotations_[d] = rot

    def _fit_tangentspace(self, X, y_enc, domains, sample_weight):
        """Fit TLRotate in tangent space.

        It computes anchors for each class of each domain.
        It computes the rotation for source domain so that its anchors
        are close to those from the target domain.
        """
        X_tgt = X[domains == self.target_domain]
        y_tgt = y_enc[domains == self.target_domain]
        weights_tgt = sample_weight[domains == self.target_domain]
        n_classes = len(np.unique(y_tgt))

        if self.n_components == "max":
            self.n_components = X.shape[1]
        self._src_pca = [
            PCA(n_components=self.n_components) for _ in range(n_classes)
        ]

        self.rotations_ = {}
        source_domains = np.unique(domains)
        source_domains = source_domains[source_domains != self.target_domain]
        for d in source_domains:
            y_src = y_enc[domains == d]
            if len(np.unique(y_src)) != n_classes:
                raise ValueError(
                    f"Number of classes in source domain {d} does not match"
                )

            # fit_transf(src) then transf(tgt), because more src data than tgt
            anchors_src, is_valid_src = self._get_anchors(
                X[domains == d],
                y_src,
                sample_weight[domains == d],
                fit_pca=True,
            )
            anchors_tgt, is_valid_tgt = self._get_anchors(
                X_tgt,
                y_tgt,
                weights_tgt,
                fit_pca=False,
            )

            if not is_valid_src or not is_valid_tgt:
                anchors_src = anchors_src[:n_classes]
                anchors_tgt = anchors_tgt[:n_classes]
                if not is_valid_src:
                    warnings.warn(f"Not enough vectors for source domain {d}")
                if not is_valid_tgt:
                    warnings.warn("Not enough vectors for target domain")

            self.rotations_[d] = _get_rotation_tangentspace(
                anchors_src,
                anchors_tgt,
                self.expl_var,
            )

    def _get_anchors(self, X, y, sample_weight, fit_pca):
        """Get anchors.

        It computes anchors for each class of each domain, ie class means and
        clusters along principal components.
        """
        is_valid = True
        classes = np.unique(y)
        anchors = np.array([
            np.average(X[y == c], axis=0, weights=sample_weight[y == c])
            for c in classes
        ])

        for i, c in enumerate(classes):
            Xc = X[y == c]
            n_vectors_c = len(Xc)

            if fit_pca:
                self._src_pca[i].fit(Xc)

            if n_vectors_c < self.n_clusters:
                is_valid = False
                break
            r = n_vectors_c / self.n_clusters

            Xc_pca = self._src_pca[i].transform(Xc)
            for j in range(self.n_components):
                inds = np.argsort(Xc_pca[:, j], axis=0)
                anchor = [
                    np.mean(Xc[inds[round(k*r):round((k+1)*r)]], axis=0)
                    for k in range(self.n_clusters)
                ]
                anchors = np.vstack([anchors, anchor])

        return anchors, is_valid

    def transform(self, X):
        """Rotate in the target domain, ie do nothing.

        .. note::
           This method is designed for using at test time on target data.
           No transformation is applied since rotations are done from source
           to target domain.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.

        Returns
        -------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Same data as in the input.
        """
        return X

    def fit_transform(self, X, y_enc, sample_weight=None):
        """Fit TLRotate and then rotate each source domain to target domain.

        It calculates and applies the rotation matrix for matching each source
        domain to the target domain.

        .. note::
           This method is designed for using at training time.
           The output for .fit_transform() will be different
           than using .fit() and .transform() separately.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.
        sample_weight : None | ndarray, shape (n_matrices,) or \
                shape (n_vectors,), default=None
            Weights for each matrix or vector. If None, it uses equal weights.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_classes)
            Set of rotated SPD matrices or tangent vectors to target domain.
        """
        self.fit(X, y_enc, sample_weight=sample_weight)
        _, _, domains = decode_domains(X, y_enc)

        X_new = np.zeros_like(X)
        for d in np.unique(domains):
            idx = domains == d

            if d == self.target_domain:
                X_new[idx] = X[idx]
            else:
                if X.ndim == 3:
                    X_new[idx] = self.rotations_[d] @ X[idx] \
                        @ self.rotations_[d].T
                else:
                    X_new[idx] = X[idx] @ self.rotations_[d]

        return X_new


###############################################################################


class TLEstimator(BaseEstimator):
    """Transfer learning wrapper for estimators.

    This is a wrapper for any BaseEstimator (classifier or regressor) that
    converts extended labels used in transfer learning into the usual y array
    to train a classifier/regressor of choice.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    estimator : BaseEstimator
        The estimator to apply on data. It can be any regressor or classifier
        from pyRiemann.
    domain_weight : None | dict, default=None
        Weights to combine data from each domain to train the estimator.
        The dict contains key=domain_name and value=weight_to_assign.
        If None, it uses equal weights.

    See Also
    --------
    TLClassifier
    TLRegressor

    Notes
    -----
    .. versionadded:: 0.4
    """

    def __init__(self, target_domain, estimator, domain_weight=None):
        """Init."""
        self.target_domain = target_domain
        self.domain_weight = domain_weight
        self.estimator = estimator

    def fit(self, X, y_enc):
        """Fit TLEstimator.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.

        Returns
        -------
        self : TLEstimator instance
            The TLEstimator instance.
        """
        if not (is_regressor(self.estimator) or is_classifier(self.estimator)):
            raise TypeError(
                "Estimator has to be either a classifier or a regressor."
            )

        X_dec, y_dec, domains = decode_domains(X, y_enc)

        if is_regressor(self.estimator):
            y_dec = y_dec.astype(float)

        if self.domain_weight is not None:
            weights = np.zeros(len(X_dec))
            for d in np.unique(domains):
                weights[domains == d] = self.domain_weight[d]
        else:
            weights = None

        if isinstance(self.estimator, Pipeline):
            sample_weight = {}
            for step in self.estimator.steps:
                step_name = step[0]
                sample_weight[step_name + "__sample_weight"] = weights
            self.estimator.fit(X_dec, y_dec, **sample_weight)
        else:
            self.estimator.fit(X_dec, y_dec, sample_weight=weights)

        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.

        Returns
        -------
        pred : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Predictions according to the estimator.
        """
        return self.estimator.predict(X)


class TLClassifier(TLEstimator):
    """Transfer learning wrapper for classifiers.

    This is a wrapper for any classifier that converts extended labels used in
    transfer learning into the usual y array to train a classifier of choice.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    estimator : BaseClassifier
        The classifier to apply on matrices.
    domain_weight : None | dict, default=None
        Weights to combine data from each domain to train the classifier.
        The dict contains key=domain_name and value=weight_to_assign.
        If None, it uses equal weights.

    See Also
    --------
    TLRegressor

    Notes
    -----
    .. versionadded:: 0.4
    """

    def fit(self, X, y_enc):
        """Fit TLClassifier.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.

        Returns
        -------
        self : TLClassifier instance
            The TLClassifier instance.
        """
        if not is_classifier(self.estimator):
            raise TypeError("Estimator has to be a classifier.")

        return super().fit(X, y_enc)

    def predict_proba(self, X):
        """Get the probability.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.

        Returns
        -------
        pred : ndarray, shape (n_matrices, n_classes) or \
                shape (n_vectors, n_classes)
            Predictions for each matrix or vector.
        """
        return self.estimator.predict_proba(X)

    def score(self, X, y_enc):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.
        """
        _, y_true, _ = decode_domains(X, y_enc)
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)


class TLRegressor(TLEstimator):
    """Transfer learning wrapper for regressors.

    This is a wrapper for any regressor that converts extended labels used in
    transfer learning into the usual y array to train a regressor of choice.

    Parameters
    ----------
    target_domain : str
        Domain to consider as target.
    estimator : BaseRegressor
        The regressor to apply on matrices.
    domain_weight : None | dict, default=None
        Weights to combine data from each domain to train the regressor.
        The dict contains key=domain_name and value=weight_to_assign.
        If None, it uses equal weights.

    See Also
    --------
    TLClassifier

    Notes
    -----
    .. versionadded:: 0.4
    """

    def fit(self, X, y_enc):
        """Fit TLRegressor.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.

        Returns
        -------
        self : TLRegressor instance
            The TLRegressor instance.
        """
        if not is_regressor(self.estimator):
            raise TypeError("Estimator has to be a regressor.")

        return super().fit(X, y_enc)

    def score(self, X, y_enc):
        """Return the coefficient of determination of the prediction.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels) or \
                shape (n_vectors, n_ts)
            Set of SPD matrices or tangent vectors.
        y_enc : ndarray, shape (n_matrices,) or shape (n_vectors,)
            Extended labels for each matrix or vector.

        Returns
        -------
        score : float
            R2 of self.predict(X) wrt. y.
        """
        _, y_true, _ = decode_domains(X, y_enc)
        y_pred = self.predict(X)
        return r2_score(y_true.astype(float), y_pred)


###############################################################################


class MDWM(MDM):
    """Classification by Minimum Distance to Weighted Mean.

    Classification by nearest centroid. For each of the given classes, a
    centroid is estimated, according to the chosen metric, as a weighted mean
    of SPD matrices from the source domain, combined with the class centroid of
    the target domain [1]_ [2]_.
    For classification, a given new matrix is attibuted to the class whose
    centroid is the nearest according to the chosen metric.

    Parameters
    ----------
    domain_tradeoff : float
        Coefficient in [0,1] controlling the transfer, ie the trade-off between
        source and target domains.
        At 0, there is no transfer, only matrices acquired from the source
        domain are used.
        At 1, this is a calibration-free system as no matrices are required
        from the source domain.
    target_domain : string
        Name of the target domain in extended labels.
    metric : string | dict, default="riemann"
        Metric used for mean estimation (for the list of supported metrics,
        see :func:`pyriemann.utils.mean.mean_covariance`) and
        for distance estimation
        (see :func:`pyriemann.utils.distance.distance`).
        The metric can be a dict with two keys, "mean" and "distance"
        in order to pass different metrics.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.

    Attributes
    ----------
    classes_ : ndarray, shape (n_classes,)
        Labels for each class.
    covmeans_ : ndarray, shape (n_classes, n_channels, n_channels)
        Centroids for each class.

    See Also
    --------
    MDM

    References
    ----------
    .. [1] `Transfer learning for SSVEP-based BCI using Riemannian similarities
        between users
        <https://hal.archives-ouvertes.fr/hal-01911092/>`_
        E. Kalunga, S. Chevallier and Q. Barthelemy, in 26th European Signal
        Processing Conference (EUSIPCO), pp. 1685-1689. IEEE, 2018.
    .. [2] `Minimizing Subject-dependent Calibration for BCI with Riemannian
        Transfer Learning
        <https://hal.archives-ouvertes.fr/hal-03202360/>`_
        S. Khazem, S. Chevallier, Q. Barthelemy, K. Haroun and C. Nous, 10th
        International IEEE/EMBS Conference on Neural Engineering (NER), pp.
        523-526. IEEE, 2021.

    Notes
    -----
    .. versionadded:: 0.4
    """

    def __init__(
        self,
        domain_tradeoff,
        target_domain,
        metric="riemann",
        n_jobs=1,
    ):
        """Init."""
        self.domain_tradeoff = domain_tradeoff
        self.target_domain = target_domain
        self.metric = metric
        self.n_jobs = n_jobs

    def fit(self, X, y_enc, sample_weight=None):
        """Fit (estimates) the centroids.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices from source and target domain.
        y_enc : ndarray, shape (n_matrices,)
            Extended labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices_source,), \
                default=None
            Weights for each matrix from the source domains.
            If None, it uses equal weights.

        Returns
        -------
        self : MDWM instance
            The MDWM instance.
        """
        self.metric_mean, self.metric_dist = check_metric(self.metric)

        if not 0 <= self.domain_tradeoff <= 1:
            raise ValueError(
                "Parameter domain_tradeoff must be included in [0, 1] "
                f"(Got {self.domain_tradeoff})"
            )

        X_dec, y_dec, domains = decode_domains(X, y_enc)
        X_src = X_dec[domains != self.target_domain]
        y_src = y_dec[domains != self.target_domain]
        X_tgt = X_dec[domains == self.target_domain]
        y_tgt = y_dec[domains == self.target_domain]

        self.classes_ = np.unique(y_src)

        if self.domain_tradeoff != 0 and set(y_tgt) != set(y_src):
            raise ValueError(
                "Classes in source domain must match classes in target domain."
                f"Classes in source are {self.classes_} while classes in "
                f"target are {np.unique(y_tgt)}"
            )

        sample_weight = check_weights(sample_weight, X_src.shape[0])

        self.source_means_ = np.stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(
                    X_src[y_src == c],
                    metric=self.metric_mean,
                    sample_weight=sample_weight[y_src == c],
                ) for c in self.classes_
            )
        )

        self.target_means_ = np.stack(
            Parallel(n_jobs=self.n_jobs)(
                delayed(mean_covariance)(
                    X_tgt[y_tgt == c],
                    metric=self.metric_mean,
                ) for c in self.classes_
            )
        )

        self.covmeans_ = geodesic(
            self.source_means_,
            self.target_means_,
            self.domain_tradeoff,
            metric=self.metric_mean,
        )

        return self

    def score(self, X, y_enc, sample_weight=None):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Test set of SPD matrices.
        y_enc : ndarray, shape (n_matrices,)
            Extended true labels for each matrix.
        sample_weight : None | ndarray, shape (n_matrices,), default=None
            Weights for each matrix. If None, it uses equal weights.

        Returns
        -------
        score : float
            Mean accuracy of clf.predict(X) wrt. y_enc.
        """
        _, y_true, _ = decode_domains(X, y_enc)
        return super().score(X, y_true, sample_weight=sample_weight)
