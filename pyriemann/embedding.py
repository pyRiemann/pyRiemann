"""Embedding covariance matrices via manifold learning techniques."""

import numpy as np
from scipy.linalg import solve, eigh
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import spectral_embedding

from .utils.kernel import kernel
from .utils.distance import pairwise_distance


class SpectralEmbedding(BaseEstimator):
    """Embed SPD matrices into an Euclidean space of smaller dimension.

    It uses Laplacian Eigenmaps [1]_ to embed SPD matrices into an Euclidean
    space. The basic hypothesis is that high-dimensional data lives in a
    low-dimensional manifold, whose intrinsic geometry can be described
    via the Laplacian matrix of a graph. The vertices of this graph are
    the SPD matrices and the weights of the links are determined by the
    Riemannian distance between each pair of them.

    Parameters
    ----------
    n_components : integer, default: 2
        The dimension of the projected subspace.
    metric : string | dict (default: 'riemann')
        The type of metric to be used for defining pairwise distance between
        covariance matrices.
    eps:  float (default: None)
        The scaling of the Gaussian kernel. If none is given
        it will use the square of the median of pairwise distances between
        points.

    References
    ----------
    .. [1] M. Belkin and P. Niyogi, "Laplacian Eigenmaps for dimensionality
        reduction and data representation," in Journal Neural Computation,
        vol. 15, no. 6, p. 1373-1396 , 2003

    """

    def __init__(self, n_components=2, metric='riemann', eps=None):
        """Init."""
        self.metric = metric
        self.n_components = n_components
        self.eps = eps

    def _get_affinity_matrix(self, X, eps):

        # make matrix with pairwise distances between points
        distmatrix = pairwise_distance(X, metric=self.metric)

        # determine which scale for the gaussian kernel
        if self.eps is None:
            eps = np.median(distmatrix)**2 / 2

        # make kernel matrix from the distance matrix
        kernel = np.exp(-distmatrix**2 / (4 * eps))

        # normalize the kernel matrix
        q = np.dot(kernel, np.ones(len(kernel)))
        kernel_n = np.divide(kernel, np.outer(q, q))

        return kernel_n

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        _check_dimensions(X,
                          n_components=self.n_components)

        affinity_matrix = self._get_affinity_matrix(X, self.eps)
        embd = spectral_embedding(adjacency=affinity_matrix,
                                  n_components=self.n_components,
                                  norm_laplacian=True)

        # normalize the embedding between -1 and +1
        embdn = 2*(embd - embd.min(0)) / embd.ptp(0) - 1

        self.embedding_ = embdn

        return self

    def fit_transform(self, X, y=None):
        """Calculate the coordinates of the embedded points.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new: ndarray, shape (n_matrices, n_components)
            Coordinates of embedded matrices.

        """
        self.fit(X)
        return self.embedding_


class LocallyLinearEmbedding(BaseEstimator, TransformerMixin):
    """Riemannian Locally Linear Embedding (LLE).

    Riemannian LLE as proposed in [1]_. LLE is a non-linear,
    neighborhood-preserving dimensionality reduction algorithm which
    consists of three main steps:
    For each point x,
    1.  find its k nearest neighbors KNN(x) and
    2.  calculate the best reconstruction of x based on its KNN.
    3.  Then calculate a low-dimensional embedding for all points based on
        the weights in step 2.

    Parameters
    ----------
    n_components : int, default: 2
        Dimensionality of projected space.
    n_neighbors : int, default: 5
        Number of neighbors for reconstruction of each point.
    metric : {'riemann'}
        Metric used for KNN and Kernel estimation.
    reg : float, default: 1e-3
        Regularization parameter.

    Attributes
    ----------
    embedding_ : ndarray, shape (n_samples, n_components)
        Stores the embedding vectors
    error_ : float
        Reconstruction error associated with `embedding_`
    data_ : ndarray, shape (n_matrices, n_channels, n_channels)
        Training data.

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] A. Goh and R. Vidal, "Clustering and dimensionality reduction
        on Riemannian manifolds", 2008 IEEE Conference on Computer Vision
        and Pattern Recognition, June 2008
    """

    def __init__(self, n_components=2, n_neighbors=5, metric='riemann',
                 reg=1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.reg = reg

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.data_ = X
        _check_dimensions(X,
                          n_components=self.n_components,
                          n_neighbors=self.n_neighbors)

        self.embedding_, self.error_ = riemann_lle(X,
                                                   self.n_components,
                                                   self.n_neighbors,
                                                   self.metric,
                                                   self.reg)
        return self

    def transform(self, X, y=None):
        """Calculate embedding coordinates for new data points based on fitted
        points.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_: ndarray, shape (n_matrices, n_components)
            Coordinates of embedded matrices.
        """
        _check_dimensions(self.data_,
                          X)
        pairwise_distances = pairwise_distance(X,
                                               self.data_,
                                               metric=self.metric)
        ind = np.array([np.argsort(dist)[1:self.n_neighbors + 1]
                        for dist in pairwise_distances])

        weights = barycenter_weights(X,
                                     self.data_,
                                     ind,
                                     metric=self.metric,
                                     reg=self.reg)

        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_new

    def fit_transform(self, X, y=None):
        """Calculate the coordinates of the embedded points.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_: ndarray, shape (n_matrices, n_components)
            Coordinates of embedded matrices.
        """
        self.fit(X)
        return self.embedding_


def barycenter_weights(X, Y, indices, metric='riemann', reg=1e-3):
    """Compute Riemannian barycenter weights of X from Y along the first axis.
    We estimate the weights to assign to each point in Y[indices] to recover
    the point X[i] by geodesic interpolation. The barycenter weights sum to 1.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    Y : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    indices : ndarray, shape (n_matrices, n_neighbors)
        Indices of the points in Y used to compute the barycenter
    metric : {'riemann'}
        Kernel metric.
    reg : float, default=1e-3
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim

    Returns
    -------
    B : ndarray, shape (n_matrices, n_neighbors)
        Interpolation weights.

    Notes
    -----
    .. versionadded:: 0.2.8
    See sklearn.manifold._locally_linear.barycenter_weights for original code.
    """

    n_matrices, n_neighbors = indices.shape
    assert X.shape[0] == n_matrices, "Number of index-sets in indices must " \
                                     "match number of matrices in X."

    B = np.empty((n_matrices, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    for i in range(n_matrices):
        X_neighbors = Y[indices[i]]
        G = kernel(X_neighbors, Cref=X[i], metric=metric)
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B


def riemann_lle(X,
                n_components=2,
                n_neighbors=5,
                metric='riemann',
                reg=1e-3):
    """Riemannian Locally Linear Embedding (LLE).

    Riemannian LLE as proposed in [1]_. LLE is a non-linear, neighborhood-
    preserving dimensionality reduction algorithm which consists of three
    main steps:
    For each point x,
    1.  find its k nearest neighbors KNN(x) and
    2.  calculate the best reconstruction of x based on its KNN.
    3.  Then calculate a low-dimensional embedding for all points based on
        the weights in step 2.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    n_components : int, default: 2
        Dimensionality of projected space.
    n_neighbors : int, default: 5
        Number of neighbors for reconstruction of each point.
    metric : {'riemann'}
        Metric used for KNN and Kernel estimation.
    reg : float, default: 1e-3
        Regularization parameter.

    Returns
    -------
    embd_ : ndarray, shape (n_matrices, n_components)
        Locally linear embedding of matrices in X.
    error_ : float
        Error of the projected embedding.

    Notes
    -----
    .. versionadded:: 0.2.8
    See sklearn.manifold._locally_linear.locally_linear_embedding for original
    code.

    References
    ----------
    .. [1] A. Goh and R. Vidal, "Clustering and dimensionality reduction
        on Riemannian manifolds", 2008 IEEE Conference on Computer Vision
        and Pattern Recognition, June 2008
    """

    n_matrices, n_channels, n_channels = X.shape
    pairwise_distances = pairwise_distance(X, metric=metric)
    neighbors = np.array([np.argsort(dist)[1:n_neighbors + 1]
                          for dist in pairwise_distances])

    B = barycenter_weights(X, X, neighbors, metric=metric, reg=reg)

    indptr = np.arange(0, n_matrices * n_neighbors + 1, n_neighbors)
    W = csr_matrix((B.ravel(), neighbors.ravel(), indptr), shape=(n_matrices,
                                                                  n_matrices))
    # M = (W - I).T * (W - I) = W.T * W - W.T - W + I
    # calculated in the two following lines
    M = (W.T * W - W.T - W).toarray()
    M.flat[:: M.shape[0] + 1] += 1

    eigen_values, eigen_vectors = eigh(
        M, eigvals=(1, n_components), overwrite_a=True
    )
    index = np.argsort(np.abs(eigen_values))
    embd_, error_ = eigen_vectors[:, index], np.sum(eigen_values)

    return embd_, error_


def _check_dimensions(X, Y=None, n_components=None, n_neighbors=None):
    n_matrices, n_channels, n_channels = X.shape

    if not isinstance(Y, type(None)):
        assert Y.shape[1:] == (n_channels, n_channels), "Dimension of " \
                                                        "matrices in data to "\
                                                        "be transformed must "\
                                                        "match dimension of " \
                                                        "data used for " \
                                                        "fitting."

    if not isinstance(n_neighbors, type(None)):
        assert n_matrices > n_neighbors, "n_neighbors must be " \
                                         "smaller than n_matrices."

    if not isinstance(n_components, type(None)):
        assert n_components < n_matrices, "n_components must be smaller " \
                                           "than n_matrices."
