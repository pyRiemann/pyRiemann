"""Embedding covariance matrices via manifold learning techniques."""

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.manifold import spectral_embedding
import umap
from .utils.distance import pairwise_distance, distance


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


class UMAP(BaseEstimator):
    """Embed SPD matrices into Euclidean space using UMAP.

    Uniform Manifold Approximation and Projection (UMAP, [1]_) is founded on three assumptions about the data:

    1. The data is uniformly distributed on Riemannian manifold;
    2. The Riemannian metric is locally constant (or can be approximated as such);
    3. The manifold is locally connected.

    From these assumptions it is possible to model the manifold with a fuzzy topological structure.
    The embedding is found by searching for a low dimensional projection of the data that has the closest possible
    equivalent fuzzy topological structure.

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
    .. [1] T. Sainburg and L. McInnes and T. Gentner, "Parametric UMAP Embeddings for Representation and
    Semisupervised Learning", in Journal Neural Computation, vol. 33, no. 11, p. 2881-2907 , 2021

    """

    def __init__(self, distance_metric='riemann', **kwargs):
        self.distance_metric = distance_metric
        self.umap_args = kwargs
        self.umapfitter = umap.UMAP(
            metric=_umap_metric_helper,
            metric_kwds={'distance_metric': self.distance_metric},
            **kwargs
        )

    def fit(self, X, y):
        """Fit the model from data in X.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            SPD matrices.
        y : ndarray shape (n_matrices, 1)
            labels corresponding to each SPD matrix.

        Returns
        -------
        self : object
            Returns the instance itself.

        """
        Xre = np.reshape(X, (len(X), -1))
        self.umapfitter.fit(Xre, y)
        return self

    def transform(self, X, y=None):
        """Calculate embedding coordinates for unseen points.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            SPD matrices.

        Returns
        -------
        X_: array-like, shape (n_matrices, n_components)
            Coordinates of embedded matrices.

        """
        Xre = np.reshape(X, (len(X), -1))
        X_ = self.umapfitter.transform(Xre)
        return X_

    def fit_transform(self, X, y=None):
        """Calculate the coordinates of the embedded points.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            SPD matrices.
        y : ndarray shape (n_matrices, 1)
            labels corresponding to each SPD matrix.

        Returns
        -------
        X_: array-like, shape (n_matrices, n_components)
            Coordinates of embedded matrices.

        """
        Xre = np.reshape(X, (len(X), -1))
        self.umapfitter.fit(Xre, y)
        X_ = self.umapfitter.transform(Xre)
        return X_


def _umap_metric_helper(A, B, distance_metric='riemann'):
    dim = int(np.sqrt(len(A)))
    A_ = np.reshape(A, (dim, dim)).astype(np.float64)  # umap casts to float32 for some reason, crashing the metric
    B_ = np.reshape(B, (dim, dim)).astype(np.float64)

    return distance(A_, B_, distance_metric)

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
