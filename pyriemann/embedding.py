"""Embedding covariance matrices via manifold learning techniques."""

import numpy as np
from scipy.linalg import solve, eigh
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import spectral_embedding

from .utils.kernel import kernel_riemann
from .utils.distance import pairwise_distance


class Embedding(BaseEstimator):
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
            ndarray of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
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
            ndarray of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new: ndarray, shape (n_matrices, n_components)
            Coordinates of embedded matrices.

        """
        self.fit(X)
        return self.embedding_


class RiemannLLE(BaseEstimator, TransformerMixin):
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
    reg : float, default: 1e-3
        Regularization parameter.
    **kwargs
        Keyword arguments passed to sklearn.manifold._locally_linear.null_space

    Attributes
    ----------
    embedding_ : ndarray, shape (n_samples, n_components)
        Stores the embedding vectors
    error_ : float
        Reconstruction error associated with `embedding_`
    data_ : int
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

    def __init__(self, n_components=2, n_neighbors=5, reg=1e-3):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.reg = reg

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        self.data_ = X
        self.embedding_, self.error_ = riemann_lle(X,
                                                   self.n_components,
                                                   self.n_neighbors,
                                                   self.reg)
        return self

    def transform(self, X, y=None):
        """Calculate embedding coordinates for new data points based on fitted
        points.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_: array-like, shape (n_matrices, n_components)
            Coordinates of embedded matrices.
        """
        pairwise_distances = pairwise_distance(X, self.data_, metric='riemann')
        ind = np.array([np.argsort(dist)[1:self.n_neighbors + 1]
                        for dist in pairwise_distances])

        weights = barycenter_weights(X, self.data_, ind, reg=self.reg)

        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding_[ind[i]].T, weights[i])
        return X_new

    def fit_transform(self, X, y=None):
        """Calculate the coordinates of the embedded points.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray | None (default: None)
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_: array-like, shape (n_matrices, n_components)
            Coordinates of embedded matrices.
        """
        self.data_ = X
        self.embedding_, self.error_ = riemann_lle(X,
                                                   self.n_components,
                                                   self.n_neighbors,
                                                   self.reg)
        return self.embedding_


def barycenter_weights(X, Y, indices, reg=1e-3):
    """Compute Riemannian barycenter weights of X from Y along the first axis.
    We estimate the weights to assign to each point in Y[indices] to recover
    the point X[i] by geodesic interpolation. The barycenter weights sum to 1.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_dim)
        ndarray of SPD matrices.
    Y : ndarray, shape (n_matrices, n_dim)
        ndarray of SPD matrices.
    indices : ndarray, shape (n_matrices, n_dim)
        Indices of the points in Y used to compute the barycenter
    reg : float, default=1e-3
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim

    Returns
    -------
    B : ndarray, shape (n_matrices, n_neighbors)

    Notes
    -----
    .. versionadded:: 0.2.8
    See sklearn.manifold._locally_linear.barycenter_weights for original code.
    """

    n_matrices, n_neighbors = indices.shape
    assert X.shape[0] == n_matrices

    B = np.empty((n_matrices, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    for i in range(len(X)):
        X_neighbors = Y[indices[i]]
        G = kernel_riemann(X_neighbors, Cref=X[i])
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
        ndarray of SPD matrices.
    n_components : int, default: 2
        Dimensionality of projected space.
    n_neighbors : int, default: 5
        Number of neighbors for reconstruction of each point.
    n_jobs : int, default: 1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel.
        If -1 all CPUs are used. If 1 is given, no parallel computing code is
        used at all, which is useful for debugging. For n_jobs below -1,
        (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
        are used.
    reg : float, default: 1e-3
        Regularization parameter.

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
    pairwise_distances = pairwise_distance(X, metric='riemann')
    neighbors = np.array([np.argsort(dist)[1:n_neighbors + 1]
                          for dist in pairwise_distances])

    B = barycenter_weights(X, X, neighbors, reg=reg)

    indptr = np.arange(0, n_matrices * n_neighbors + 1, n_neighbors)
    W = csr_matrix((B.ravel(), neighbors.ravel(), indptr), shape=(n_matrices,
                                                                  n_matrices))
    # M = (W - I).T * (W - I) = W.T * W - W.T - W + I
    M = (W.T * W - W.T - W).toarray()
    M.flat[:: M.shape[0] + 1] += 1
    return null_space(
        M,
        n_components,
        k_skip=1,
    )


def null_space(M, k, k_skip=1):
    """
    Find the null space of a matrix M.

    Parameters
    ----------
    M : ndarray
        Input covariance matrix: should be symmetric positive semi-definite
    k : int
        Number of eigenvalues/vectors to return
    k_skip : int, default=1
        Number of low eigenvalues to skip.

    Notes
    -----
    .. versionadded:: 0.2.8
    See sklearn.manifold._locally_linear.null_space for original code.
    """

    if hasattr(M, "toarray"):
        M = M.toarray()
    eigen_values, eigen_vectors = eigh(
        M, eigvals=(k_skip, k + k_skip - 1), overwrite_a=True
    )
    index = np.argsort(np.abs(eigen_values))
    return eigen_vectors[:, index], np.sum(eigen_values)
