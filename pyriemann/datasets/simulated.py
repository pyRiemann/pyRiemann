import numpy as np
from sklearn.utils.validation import check_random_state

from ..utils.distance import distance_riemann
from ..utils.base import powm, sqrtm
from .sampling import generate_random_spd_matrix, sample_gaussian_spd


def make_covariances(n_matrices, n_channels, rs, return_params=False,
                     evals_mean=2.0, evals_std=0.1):
    """Generate a set of covariances matrices, with the same eigenvectors.

    Parameters
    ----------
    n_matrices : int
        Number of matrices to generate.
    n_channels : int
        Number of channels in covariance matrices.
    rs : RandomState instance
        Random state for reproducible output across multiple function calls.
    return_params : bool (default False)
        If True, then return parameters.
    evals_mean : float (default 2.0)
        Mean of eigen values.
    evals_std : float (default 0.1)
        Standard deviation of eigen values.

    Returns
    -------
    covmats : ndarray, shape (n_matrices, n_channels, n_channels)
        Covariances matrices.
    evals : ndarray, shape (n_matrices, n_channels)
        Eigen values used for each covariance matrix.
        Only returned if ``return_params=True``.
    evecs : ndarray, shape (n_channels, n_channels)
        Eigen vectors used for all covariance matrices.
        Only returned if ``return_params=True``.
    """
    evals = np.abs(evals_mean + evals_std * rs.randn(n_matrices, n_channels))
    evecs, _ = np.linalg.qr(rs.randn(n_channels, n_channels))

    covmats = np.empty((n_matrices, n_channels, n_channels))
    for i in range(n_matrices):
        covmats[i] = evecs @ np.diag(evals[i]) @ evecs.T

    if return_params:
        return covmats, evals, evecs
    else:
        return covmats


def make_masks(n_masks, n_dim0, n_dim1_min, rs):
    """Generate a set of masks, defined as semi-orthogonal matrices.

    Parameters
    ----------
    n_masks : int
        Number of masks to generate.
    n_dim0 : int
        First dimension of masks.
    n_dim1_min : int
        Minimal value for second dimension of masks.
    rs : RandomState instance
        Random state for reproducible output across multiple function calls.

    Returns
    -------
    masks : list of n_masks ndarray of shape (n_dim0, n_dim1_i), \
            with different n_dim1_i, such that n_dim1_min <= n_dim1_i <= n_dim0
        Masks.
    """
    masks = []
    for i in range(n_masks):
        n_dim1 = rs.randint(n_dim1_min, n_dim0, size=1)[0]
        mask, _ = np.linalg.qr(rs.randn(n_dim0, n_dim1))
        masks.append(mask)
    return masks


def make_gaussian_blobs(n_matrices=100, n_dim=2, class_sep=1.0, class_disp=1.0,
                        return_centers=False, random_state=None, *,
                        mat_mean=.0, mat_std=1.):
    """Generate SPD dataset with two classes sampled from Riemannian Gaussian.

    Generate a dataset with SPD matrices drawn from two Riemannian Gaussian
    distributions. The distributions have the same class dispersions and the
    distance between their centers of mass is an input parameter. Useful for
    testing classification or clustering methods.

    Parameters
    ----------
    n_matrices : int (default: 100)
        How many matrices to generate for each class.
    n_dim : int (default: 2)
        Dimensionality of the SPD matrices generated by the distributions.
    class_sep : float (default: 1.0)
        Parameter controlling the separability of the classes.
    class_disp : float (default: 1.0)
        Intra dispersion of the points sampled from each class.
    return_centers : bool (default: False)
        If True, then return the centers of each cluster
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.
    mat_mean : float (default 0.0)
        Mean of random values to generate matrices.
    mat_std : float (default 1.0)
        Standard deviation of random values to generate matrices.

    Returns
    -------
    X : ndarray, shape (2*n_matrices, n_dim, n_dim)
        Set of SPD matrices.
    y : ndarray, shape (2*n_matrices,)
        Labels corresponding to each matrix.
    centers : ndarray, shape (2, n_dim, n_dim)
        The centers of each class. Only returned if ``return_centers=True``.

    Notes
    -----
    .. versionadded:: 0.2.8

    """
    if not isinstance(class_sep, float):
        raise ValueError(f'class_sep must be a float (Got {class_sep})')

    rs = check_random_state(random_state)

    # generate dataset for class 0
    C0 = generate_random_spd_matrix(
        n_dim,
        random_state=random_state,
        mat_mean=mat_mean,
        mat_std=mat_std
    )
    X0 = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=C0,
        sigma=class_disp,
        random_state=random_state
    )
    y0 = np.zeros(n_matrices)

    # generate dataset for class 1
    epsilon = np.exp(class_sep / np.sqrt(n_dim))
    C1 = epsilon * C0
    X1 = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=C1,
        sigma=class_disp,
        random_state=random_state
    )
    y1 = np.ones(n_matrices)

    X = np.concatenate([X0, X1])
    y = np.concatenate([y0, y1])
    idx = rs.permutation(len(X))
    X = X[idx]
    y = y[idx]

    if return_centers:
        centers = np.stack([C0, C1])
        return X, y, centers
    else:
        return X, y


def make_outliers(n_matrices, mean, sigma, outlier_coeff=10,
                  random_state=None):
    """Generate a set of outlier points.

    Simulate data points that are outliers for a given Riemannian Gaussian
    distribution with fixed mean and dispersion.

    Parameters
    ----------
    n_matrices : int
        How many matrices to generate.
    mean : ndarray, shape (n_dim, n_dim)
        Center of the Riemannian Gaussian distribution.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    outlier_coeff: float
        Coefficient determining how to define an outlier data point, i.e. how
        many times the sigma parameter its distance to the mean should be.
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    outliers : ndarray, shape (n_matrices, n_dim, n_dim)
        Set of simulated outlier matrix.

    Notes
    -----
    .. versionadded:: 0.2.8
    """

    n_dim = mean.shape[1]
    mean_sqrt = sqrtm(mean)

    outliers = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Oi = generate_random_spd_matrix(n_dim=n_dim, random_state=random_state)
        epsilon_num = outlier_coeff * sigma * n_dim
        epsilon_den = distance_riemann(Oi, np.eye(n_dim))**2
        epsilon = np.sqrt(epsilon_num / epsilon_den)
        outliers[i] = mean_sqrt @ powm(Oi, epsilon) @ mean_sqrt

    return outliers
