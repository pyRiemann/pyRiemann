"""Embedding covariance matrices via manifold learning techniques."""

import numpy as np
from scipy.linalg import solve
from scipy.sparse import csr_matrix

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import spectral_embedding
from sklearn.manifold._locally_linear import null_space

from .utils.mean import mean_riemann
from .utils.base import logm, invsqrtm
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
            SPD matrices.

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
            SPD matrices.

        Returns
        -------
        X_new: array-like, shape (n_matrices, n_components)
            Coordinates of embedded matrices.

        """
        self.fit(X)
        return self.embedding_


class RiemannLLE(BaseEstimator, TransformerMixin):
    """
    Wrapper for UMAP for dimensionality reduction on positive definite matrices with a Riemannian metric
    Parameters
    ----------
    metric : str (default 'riemann')
        string code for the metric from .utils.distance
    **kwargs : dict
        arguments to pass to umap.
    """

    def __init__(self, n_components=2, n_neighbors=5, n_jobs=1, reg=1e-3, **kwargs):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs
        self.reg = reg
        self.null_space_args = kwargs

    def fit(self, X, y=None):
        self.data = X
        self.embedding, self.reconstruction_error = riemann_lle(X, self.n_components, self.n_neighbors, self.reg,
                                                                self.null_space_args)
        return self

    def transform(self, X, y=None):
        pairwise_distances = pairwise_distance(X, self.data, metric='riemann')
        ind = np.array([np.argsort(dist)[1:self.n_neighbors + 1] for dist in pairwise_distances])

        weights = barycenter_weights(X, self.data, ind, reg=self.reg)

        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = np.dot(self.embedding[ind[i]].T, weights[i])
        return X_new

    def fit_transform(self, X, y=None):
        self.data = X
        self.embedding, self.reconstruction_error = riemann_lle(X, self.n_components, self.n_neighbors, self.reg,
                                                                self.null_space_args)
        return self.embedding


def barycenter_weights(X, Y, indices, reg=1e-3):
    """Compute Riemannian barycenter weights of X from Y along the first axis.
    We estimate the weights to assign to each point in Y[indices] to recover
    the point X[i] by geodesic interpolation. The barycenter weights sum to 1.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_dim)
    Y : array-like, shape (n_samples, n_dim)
    indices : array-like, shape (n_samples, n_dim)
            Indices of the points in Y used to compute the barycenter
    reg : float, default=1e-3
        amount of regularization to add for the problem to be
        well-posed in the case of n_neighbors > n_dim
    Returns
    -------
    B : array-like, shape (n_samples, n_neighbors)
    Notes
    -----
    See developers note for more information.
    """
    # indices = check_array(indices, dtype=int)

    n_samples, n_neighbors = indices.shape
    assert X.shape[0] == n_samples

    B = np.empty((n_samples, n_neighbors), dtype=X.dtype)
    v = np.ones(n_neighbors, dtype=X.dtype)

    for i in range(len(X)):
        X_neighbors = Y[indices[i]]
        G = riemann_kernel_matrix(X_neighbors, X_neighbors, X[i])
        trace = np.trace(G)
        if trace > 0:
            R = reg * trace
        else:
            R = reg
        G.flat[:: n_neighbors + 1] += R
        w = solve(G, v, sym_pos=True)
        B[i, :] = w / np.sum(w)
    return B


def riemann_lle(X, n_components=2, n_neighbors=5, reg=1e-3, null_space_args={}):
    n_samples = X.shape[0]
    pairwise_distances = pairwise_distance(X, metric='riemann')
    neighbors = np.array([np.argsort(dist)[1:n_neighbors + 1] for dist in pairwise_distances])

    B = barycenter_weights(X, X, neighbors, reg=reg)

    indptr = np.arange(0, n_samples * n_neighbors + 1, n_neighbors)
    W = csr_matrix((B.ravel(), neighbors.ravel(), indptr), shape=(n_samples, n_samples))
    M = (W.T * W - W.T - W).toarray()
    M.flat[:: M.shape[0] + 1] += 1  # W = W - I = W - I
    return null_space(
        M,
        n_components,
        k_skip=1,
        **null_space_args
    )


def riemann_kernel_matrix(X, Y=None, Cref=None):
    if Cref is None:
        G = mean_riemann(X)
        G_invsq = invsqrtm(G)

    else:
        G_invsq = invsqrtm(Cref)

    Ntx, Ne, Ne = X.shape

    X_ = np.zeros((Ntx, Ne, Ne))
    for index in range(Ntx):
        X_[index] = logm(G_invsq @ X[index] @ G_invsq)

    if Y is None:
        Nty, Ne, Ne = X.shape
        Y_ = X_

    else:
        Nty, Ne, Ne = Y.shape
        Y_ = np.zeros((Nty, Ne, Ne))
        for index in range(Nty):
            Y_[index] = logm(G_invsq @ Y[index] @ G_invsq)

    res = np.zeros((Nty, Ntx))
    for i in range(Nty):
        for j in range(Ntx):
            res[i][j] = np.trace(X_[i] @ Y_[j])
    return res
