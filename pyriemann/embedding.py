"""Embedding SPD matrices via manifold learning techniques."""

import warnings

import numpy as np
from scipy.linalg import solve, eigh
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.manifold import spectral_embedding
from sklearn.manifold._utils import _binary_search_perplexity

from .utils.distance import pairwise_distance
from .utils.kernel import kernel as kernel_fct
from .optimization.positive_definite import _run_minimization
from .optimization.positive_definite import _get_initial_solution


class SpectralEmbedding(BaseEstimator):
    """Spectral embedding of SPD/HPD matrices.

    Spectral embedding uses Laplacian Eigenmaps [1]_ to embed SPD/HPD matrices
    into an Euclidean space of smaller dimension.
    The basic hypothesis is that high-dimensional
    data live in a low-dimensional manifold, whose intrinsic geometry can be
    described via the Laplacian matrix of a graph. The vertices of this graph
    are the SPD/HPD matrices and the weights of the links are determined by the
    Riemannian distance between each pair of them.

    Parameters
    ----------
    n_components : integer, default=2
        The dimension of the projected subspace.
    metric : string, default="riemann"
        Metric used for defining pairwise distance between SPD/HPD matrices.
        For the list of supported metrics,
        see :func:`pyriemann.utils.distance.pairwise_distance`.
    eps : None | float, default=None
        The scaling of the Gaussian kernel. If none is given it will use the
        square of the median of pairwise distances between matrices.

    Attributes
    ----------
    embedding_ : ndarray, shape (n_matrices, n_components)
        Embedding vectors of the training set.

    References
    ----------
    .. [1] `Laplacian Eigenmaps for dimensionality
        reduction and data representation
        <https://ieeexplore.ieee.org/document/6789755>`_
        M. Belkin and P. Niyogi, in Neural Computation, vol. 15, no. 6,
        p. 1373-1396 , 2003
    """

    def __init__(self, n_components=2, metric="riemann", eps=None):
        """Init."""
        self.metric = metric
        self.n_components = n_components
        self.eps = eps

    def _get_affinity_matrix(self, X):

        # make matrix with pairwise distances between matrices
        distmatrix = pairwise_distance(X, metric=self.metric)

        # determine which scale for the gaussian kernel
        if self.eps is None:
            eps = np.median(distmatrix) ** 2 / 2
        else:
            eps = self.eps

        # make kernel matrix from the distance matrix
        kernel = np.exp(-(distmatrix**2) / (4 * eps))

        # normalize the kernel matrix
        q = kernel @ np.ones(len(kernel))
        kernel_n = np.divide(kernel, np.outer(q, q))

        return kernel_n

    def fit(self, X, y=None):
        """Fit the spectral embedding.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : SpectralEmbedding instance
            The SpectralEmbedding instance.
        """
        _check_dimensions(X, n_components=self.n_components)

        affinity_matrix = self._get_affinity_matrix(X)
        embd = spectral_embedding(
            adjacency=affinity_matrix,
            n_components=self.n_components,
            norm_laplacian=True,
        )

        # normalize the embedding between -1 and +1
        embdn = 2 * (embd - embd.min(0)) / np.ptp(embd, 0) - 1
        self.embedding_ = embdn

        return self

    def fit_transform(self, X, y=None):
        """Calculate the coordinates of the embedded matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_components)
            Coordinates of embedded matrices.
        """
        self.fit(X)
        return self.embedding_


class LocallyLinearEmbedding(TransformerMixin, BaseEstimator):
    """Locally Linear Embedding of SPD matrices.

    Locally Linear Embedding (LLE) is a non-linear,
    neighborhood-preserving dimensionality reduction algorithm which
    consists of three main steps [1]_.
    For each SPD matrix X[i] [2]_:

    1.  find its k-nearest neighbors k-NN(X[i]),
    2.  calculate the best reconstruction of X[i] based on its k-NN,
    3.  calculate a low-dimensional embedding for all matrices based on
        the weights in step 2.

    Parameters
    ----------
    n_components : int | None, default=2
        Dimensionality of projected space.
        If None, ``n_components`` is set to ``n_matrices - 1``.
    n_neighbors : int | None, default=5
        Number of neighbors for reconstruction of each matrix.
        If None, all available matrices are used.
        If ``n_neighbors > n_matrices``, ``n_neighbors`` is set to
        ``n_matrices - 1``.
    metric : string, default="riemann"
        Metric used for k-NN and kernel estimation. For the list of supported
        metrics, see :func:`pyriemann.utils.kernel.kernel`.
    kernel : callable | None, default=None
        Kernel function to use for the embedding. If None, the canonical
        kernel specified by the metric is used. Must be a function that
        takes the arguments (X, Cref, metric).
    reg : float, default=1e-3
        Regularization parameter.

    Attributes
    ----------
    embedding_ : ndarray, shape (n_matrices, n_components)
        Embedding vectors of the training set.
    error_ : float
        Reconstruction error associated with `embedding_`.
    data_ : ndarray, shape (n_matrices, n_channels, n_channels)
        Training set.

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `Nonlinear Dimensionality Reduction by Locally Linear Embedding
        <https://www.science.org/doi/10.1126/science.290.5500.2323>`_
        S. Roweis and L. K. Saul, in Science, Vol 290, Issue 5500, pp.
        2323-2326, 2000.
    .. [2] `Clustering and dimensionality reduction on Riemannian manifolds
        <https://ieeexplore.ieee.org/document/4587422>`_
        A. Goh and R. Vidal, in 2008 IEEE Conference on Computer Vision and
        Pattern Recognition
    """

    def __init__(
        self,
        n_components=2,
        n_neighbors=5,
        metric="riemann",
        kernel=None,
        reg=1e-3,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.reg = reg
        self.kernel = kernel

    def fit(self, X, y=None):
        """Fit the model from X.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : LocallyLinearEmbedding instance
            The LocallyLinearEmbedding instance.
        """
        self.data_ = X
        self.n_components, self.n_neighbors = _check_dimensions(
            X,
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
        )

        self.embedding_, self.error_ = locally_linear_embedding(
            X,
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            reg=self.reg,
            kernel=self.kernel,
        )

        return self

    def transform(self, X, y=None):
        """Calculate embedding coordinates.

        Calculate embedding coordinates for new matrices based on fitted
        matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_components)
            Coordinates of embedded matrices.
        """
        _check_dimensions(self.data_, X)
        pairwise_dists = pairwise_distance(X, self.data_, metric=self.metric)
        ind = np.array(
            [np.argsort(dist)[1: self.n_neighbors + 1]
             for dist in pairwise_dists]
        )

        weights = barycenter_weights(
            X,
            self.data_,
            ind,
            metric=self.metric,
            reg=self.reg,
            kernel=self.kernel,
        )

        X_new = np.empty((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_new[i] = self.embedding_[ind[i]].T @ weights[i]
        return X_new

    def fit_transform(self, X, y=None):
        """Fit and calculate the coordinates of the embedded matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_components)
            Coordinates of embedded matrices.
        """
        self.fit(X)
        return self.embedding_


def barycenter_weights(X, Y, indices, metric="riemann", kernel=None, reg=1e-3):
    """Compute barycenter weights of X from Y along the first axis.

    Estimates the weights to assign to each matrix in Y[indices] to recover
    the matrix X[i] by geodesic interpolation. The barycenter weights sum to 1.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    Y : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    indices : ndarray, shape (n_matrices, n_neighbors)
        Indices of matrices in Y used to compute the barycenter.
    metric : string, default="riemann"
        Kernel metric. For the list of supported metrics, see
        :func:`pyriemann.utils.kernel.kernel`.
    kernel : callable | None, default=None
        Kernel function to use for the embedding. If None, the canonical
        kernel specified by the metric is used. Must be a function that
        takes the arguments (X, Cref, metric).
    reg : float, default=1e-3
        Amount of regularization to add for the problem to be
        well-posed in the case of ``n_neighbors > n_channels``.

    Returns
    -------
    B : ndarray, shape (n_matrices, n_neighbors)
        Interpolation weights.

    Notes
    -----
    .. versionadded:: 0.3
    """
    n_matrices, n_neighbors = indices.shape
    msg = (
        f"Number of index-sets in indices (is {n_matrices}) must match "
        f"number of matrices in X (is {X.shape[0]})."
    )
    assert X.shape[0] == n_matrices, msg
    if kernel is None:
        kernel = kernel_fct
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
        w = solve(G, v, assume_a="pos")
        B[i] = w / np.sum(w)
    return B


def locally_linear_embedding(
    X,
    *,
    n_components=2,
    n_neighbors=5,
    metric="riemann",
    kernel=None,
    reg=1e-3,
):
    """Perform a Locally Linear Embedding (LLE) of SPD matrices.

    Locally Linear Embedding (LLE) is a non-linear,
    neighborhood-preserving dimensionality reduction algorithm which consists
    of three main steps [1]_. For each SPD matrix X[i]:

    1.  find its k-nearest neighbors k-NN(X[i]),
    2.  calculate the best reconstruction of X[i] based on its
        k-nearest neighbors (Eq.9 in [1]_),
    3.  calculate a low-dimensional embedding for all matrices based on
        the weights in step 2.

    Parameters
    ----------
    X : ndarray, shape (n_matrices, n_channels, n_channels)
        Set of SPD matrices.
    n_components : int, default=2
        Dimensionality of projected space.
        If None, ``n_components`` is set to ``n_matrices - 1``.
    n_neighbors : int, default=5
        Number of neighbors for reconstruction of each matrix.
        If None, all available matrices are used.
        If ``n_neighbors > n_matrices``, ``n_neighbors`` is set to
        ``n_matrices - 1``.
    metric : string, default="riemann"
        Metric used for k-NN and kernel estimation. For the list of supported
        metrics, see :func:`pyriemann.utils.kernel.kernel`.
    kernel : callable | None, default=None
        Kernel function to use for the embedding. If None, the canonical
        kernel specified by the metric is used. Must be a function that
        takes the arguments (X, Cref, metric).
    reg : float, default=1e-3
        Regularization parameter.

    Returns
    -------
    embd : ndarray, shape (n_matrices, n_components)
        Locally linear embedding of matrices in X.
    error : float
        Error of the projected embedding.

    Notes
    -----
    .. versionadded:: 0.3

    References
    ----------
    .. [1] `Clustering and dimensionality reduction on Riemannian manifolds
        <https://ieeexplore.ieee.org/document/4587422>`_
        A. Goh and R. Vidal, in 2008 IEEE Conference on Computer Vision and
        Pattern Recognition
    """
    n_matrices, n_channels, n_channels = X.shape
    pairwise_distances = pairwise_distance(X, metric=metric)
    neighbors = np.array(
        [np.argsort(dist)[1: n_neighbors + 1] for dist in pairwise_distances]
    )

    B = barycenter_weights(
        X,
        X,
        neighbors,
        metric=metric,
        reg=reg,
        kernel=kernel,
    )

    indptr = np.arange(0, n_matrices * n_neighbors + 1, n_neighbors)
    W = csr_matrix(
        (B.ravel(), neighbors.ravel(), indptr),
        shape=(n_matrices, n_matrices),
    )
    # M = (W - I).T * (W - I) = W.T * W - W.T - W + I
    # calculated in the two following lines
    M = (W.T * W - W.T - W).toarray()
    M.flat[:: M.shape[0] + 1] += 1

    eigen_values, eigen_vectors = eigh(
        M, subset_by_index=(1, n_components), overwrite_a=True
    )
    index = np.argsort(np.abs(eigen_values))
    embd, error = eigen_vectors[:, index], np.sum(eigen_values)

    return embd, error


class TSNE(BaseEstimator):
    """T-distributed Stochastic Neighbor Embedding (t-SNE) of SPD/HPD matrices.

    T-distributed Stochastic Neighbor Embedding (t-SNE) reduces
    a set of nxn SPD/HPD matrices into a set of 2x2 SPD/HPD matrices [1]_.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the matrices in the embedded space.
    perplexity : int, default=None
        Perplexity used in the t-SNE algorithm.
        If None, it will be set to 0.75*n_matrices.
    metric : {"logeuclid", "riemann"}, default="riemann"
        Metric for the gradient descent.
    verbosity : int, default=0
        Level of information printed by the optimizer while it operates:
        0 is silent, 2 is most verbose.
    max_iter : int, default=10_000
        Maximum number of iterations used for the gradient descent.
    max_time : int default=300
        Maximum time on the run time of the gradient descent in seconds.
    random_state : int, default=None
        Pass an int for reproducible output across multiple function calls.

    Attributes
    ----------
    embedding_ : ndarray, shape (n_matrices, n_components, n_components)
        Embedding matrices of the training set.

    References
    ----------
    .. [1] `Geometry-Aware visualization of high dimensional Symmetric
        Positive Definite matrices
        <https://openreview.net/pdf?id=DYCSRf3vby>`_
        T. de Surrel, S. Chevallier, F. Lotte and F. Yger.
        Transactions on Machine Learning Research, 2025
    """

    def __init__(
        self,
        n_components=2,
        perplexity=None,
        metric="riemann",
        verbosity=0,
        max_iter=10000,
        max_time=300,
        random_state=None,
    ):
        self.n_components = n_components
        self.perplexity = perplexity
        self.metric = metric
        self.verbosity = verbosity
        self.max_iter = max_iter
        self.max_time = max_time
        self.random_state = random_state

    def _compute_similarities(self, X):
        """Compute the high dimensional symmetrized conditional similarities
        p_{ij} for t-SNE.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices to reduce.

        Returns
        -------
        P : ndarray, shape (n_matrices, n_matrices)
            Symmetrized conditional probabilities of X.
        """
        n_matrices, _, _ = X.shape
        Dsq = pairwise_distance(X, metric=self.metric, squared=True)
        Dsq = Dsq.astype(np.float32, copy=False)
        # Use _binary_search_perplexity from sklearn to compute conditional
        # probabilities such that they approximately match the desired
        # perplexity
        conditional_P = _binary_search_perplexity(Dsq, self.perplexity, 0)

        # Symmetrize the conditional probabilities
        P = conditional_P + conditional_P.T
        return P / (2 * n_matrices)

    def _compute_low_affinities(self, Y):
        """Compute the low dimensional similarities q_{ij} for t-SNE.

        Parameters
        ----------
        Y : ndarray, shape (n_matrices, n_components, n_components)
            Set of SPD matrices.

        Returns
        -------
        Q : ndarray, shape (n_matrices, n_matrices)
            Low dimensional similarities conditional probabilities of Y.
        Dsq : ndarray, shape (n_matrices, n_matrices)
            Squared distances between the matrices in Y.
        """
        n_matrices, _, _ = Y.shape
        Dsq = pairwise_distance(Y, metric=self.metric, squared=True)

        denominator = np.sum(
            [np.sum([np.delete(1 / (1 + Dsq[k, :]), k)])
             for k in range(n_matrices)]
        )
        Q = 1 / (1 + Dsq) / denominator
        np.fill_diagonal(Q, 0)
        return Q, Dsq

    def fit(self, X, y=None):
        """Fit TSNE.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        self : TSNE instance
            The TSNE instance.
        """

        if self.metric not in ["riemann", "logeuclid"]:
            raise ValueError(
                f"Unknown metric '{self.metric}'. "
                f"Use 'riemann' or 'logeuclid'."
            )

        n_matrices, _, _ = X.shape

        if self.perplexity is None:
            self.perplexity = int(0.75 * n_matrices)

        # Compute similarities in the high dimension space
        P = self._compute_similarities(X)

        # Sample initial solution close to the identity
        initial_point = _get_initial_solution(
            n_matrices, self.n_components, self.random_state
        )
        if self.verbosity >= 1:
            print("Optimizing...")
        self.embedding_ = _run_minimization(
            P,
            initial_point,
            self.metric,
            self.max_iter,
            self.max_time,
            self.verbosity,
            self._compute_low_affinities
        )

        return self

    def fit_transform(self, X, y=None):
        """Calculate the coordinates of the embedded matrices.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD/HPD matrices.
        y : None
            Not used, here for compatibility with sklearn API.

        Returns
        -------
        X_new : ndarray, shape (n_matrices, n_components, n_components)
            Coordinates of embedded matrices.
        """
        self.fit(X)
        return self.embedding_


def _check_dimensions(X, Y=None, n_components=None, n_neighbors=None):
    n_matrices, n_channels, n_channels = X.shape

    if Y is not None and Y.shape[1:] != (n_channels, n_channels):
        msg = (
            f"Dimension of matrices in data to be transformed must match "
            f"dimension of data used for fitting. Expected "
            f"{(n_channels, n_channels)}, got {Y.shape[1:]}."
        )
        raise ValueError(msg)

    if n_components is None:
        n_components = n_matrices - 1
    elif n_components >= n_matrices:
        msg = (
            f"n_components (is {n_components}) must be smaller than "
            f"n_matrices (is {n_matrices})."
        )
        raise ValueError(msg)

    if n_neighbors is None:
        n_neighbors = n_matrices - 1
    elif n_matrices <= n_neighbors:
        warnings.warn(
            f"n_neighbors (is {n_neighbors}) must be smaller than "
            f"n_matrices (is {n_matrices}). Setting n_neighbors to "
            f"{n_matrices - 1}."
        )
        n_neighbors = n_matrices - 1

    return n_components, n_neighbors
