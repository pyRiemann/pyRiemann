import numpy as np
from sklearn.utils.validation import check_random_state

from ..utils.mean import mean_riemann
from ..utils.distance import distance_riemann
from ..utils.base import invsqrtm, powm, sqrtm, expm
from .sampling import generate_random_spd_matrix, sample_gaussian_spd
from ..transfer import encode_domains
from ..utils import deprecated


@deprecated(
    "make_covariances is deprecated and will be removed in 0.6.0; "
    "please use make_matrices."
)
def make_covariances(n_matrices, n_channels, rs=None, return_params=False,
                     evals_mean=2.0, evals_std=0.1):
    """Generate a set of covariances matrices, with the same eigenvectors.

    Parameters
    ----------
    n_matrices : int
        Number of matrices to generate.
    n_channels : int
        Number of channels in covariance matrices.
    rs : RandomState instance, default=None
        Random state for reproducible output across multiple function calls.
    return_params : bool, default=False
        If True, then return parameters.
    evals_mean : float, default=2.0
        Mean of eigen values.
    evals_std : float, default=0.1
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
    rs = check_random_state(rs)

    evals = np.abs(evals_mean + evals_std * rs.randn(n_matrices, n_channels))
    evecs, _ = np.linalg.qr(rs.randn(n_channels, n_channels))

    covmats = np.empty((n_matrices, n_channels, n_channels))
    for i in range(n_matrices):
        covmats[i] = evecs @ np.diag(evals[i]) @ evecs.T

    if return_params:
        return covmats, evals, evecs
    else:
        return covmats


def make_matrices(n_matrices, n_dim, kind, rs=None, return_params=False,
                  evals_low=0.5, evals_high=2.0, eigvecs_same=False):
    """Generate a set of matrices, with specific properties.

    Parameters
    ----------
    n_matrices : int
        Number of matrices to generate.
    n_dim : int
        Dimension of square matrices to generate.
    kind : {'real', 'comp', 'spd', 'spsd', 'hpd', 'hpsd'}
        Kind of matrices to generate:

        - 'real' for real-valued matrices;
        - 'comp' for complex-valued matrices;
        - 'spd' for symmetric positive-definite matrices;
        - 'spsd' for symmetric positive semi-definite matrices;
        - 'hpd' for Hermitian positive-definite matrices;
        - 'hpsd' for Hermitian positive semi-definite matrices.
    rs : RandomState instance, default=None
        Random state for reproducible output across multiple function calls.
    return_params : bool, default=False
        If True, then returns evals and evecs for 'spd', 'spsd', 'hpd' and
        'hpsd'.
    evals_low : float, default=0.5
        Lowest value of the uniform distribution to draw eigen values.
    evals_high : float, default=2.0
        Highest value of the uniform distribution to draw eigen values.
    eigvecs_same : bool, default False
        If True, then uses the same eigen vectors for all matrices.

    Returns
    -------
    mats : ndarray, shape (n_matrices, n_dim, n_dim)
        Generated matrices.
    evals : ndarray, shape (n_matrices, n_dim)
        Eigen values used for 'spd', 'spsd', 'hpd' and 'hpsd'.
        Only returned if ``return_params=True``.
    evecs : ndarray, shape (n_matrices, n_dim, n_dim) or (n_dim, n_dim)
        Eigen vectors used for 'spd', 'spsd', 'hpd' and 'hpsd'.
        Only returned if ``return_params=True``.

    Notes
    -----
    .. versionadded:: 0.5
    """
    if kind not in ("real", "comp", "spd", "spsd", "hpd", "hpsd"):
        raise ValueError(f"Unsupported matrix kind: {kind}")

    rs = check_random_state(rs)
    X = rs.randn(n_matrices, n_dim, n_dim)
    if kind == "real":
        return X

    if kind in ("comp", "hpd", "hpsd"):
        X = X + 1j * rs.randn(n_matrices, n_dim, n_dim)
        if kind == "comp":
            return X

    # eigen values
    if evals_low <= 0.0:
        raise ValueError(
            f"Lowest value must be strictly positive (Got {evals_low}).")
    if evals_high <= evals_low:
        raise ValueError(
            "Highest value must be superior to lowest value "
            f"(Got {evals_high} and {evals_low}).")
    evals = rs.uniform(evals_low, evals_high, size=(n_matrices, n_dim))
    if kind in ("spsd", "hpsd"):
        evals[..., -1] = 1e-10  # last eigen value set to almost zero

    # eigen vectors
    if eigvecs_same:
        X = X[0]
    if np.__version__ < '1.22.0' and X.ndim > 2:
        evecs = np.array([np.linalg.qr(x)[0] for x in X])
    else:
        evecs = np.linalg.qr(X)[0]

    # conjugation
    if eigvecs_same:
        mats = np.empty((n_matrices, n_dim, n_dim), dtype=X.dtype)
        for i in range(n_matrices):
            mats[i] = (evecs * evals[i]) @ evecs.conj().T
    else:
        mats = (evecs * evals[:, np.newaxis, :]) @ np.swapaxes(evecs.conj(),
                                                               -2, -1)

    if return_params:
        return mats, evals, evecs
    else:
        return mats


def make_masks(n_masks, n_dim0, n_dim1_min, rs=None):
    """Generate a set of masks, defined as semi-orthogonal matrices.

    Parameters
    ----------
    n_masks : int
        Number of masks to generate.
    n_dim0 : int
        First dimension of masks.
    n_dim1_min : int
        Minimal value for second dimension of masks.
    rs : RandomState instance, default=None
        Random state for reproducible output across multiple function calls.

    Returns
    -------
    masks : list of n_masks ndarray of shape (n_dim0, n_dim1_i), \
            with different n_dim1_i, such that n_dim1_min <= n_dim1_i <= n_dim0
        Masks.

    Notes
    -----
    .. versionadded:: 0.3
    """
    rs = check_random_state(rs)

    masks = []
    for _ in range(n_masks):
        n_dim1 = rs.randint(n_dim1_min, n_dim0, size=1)[0]
        mask, _ = np.linalg.qr(rs.randn(n_dim0, n_dim1))
        masks.append(mask)
    return masks


def make_gaussian_blobs(n_matrices=100, n_dim=2, class_sep=1.0, class_disp=1.0,
                        return_centers=False, center_dataset=False,
                        random_state=None, centers=None, *, n_jobs=1,
                        sampling_method='auto'):
    """Generate SPD dataset with two classes sampled from Riemannian Gaussian.

    Generate a dataset with SPD matrices drawn from two Riemannian Gaussian
    distributions. The distributions have the same class dispersions and the
    distance between their centers of mass is an input parameter. Useful for
    testing classification or clustering methods.

    Parameters
    ----------
    n_matrices : int, default=100
        How many matrices to generate for each class.
    n_dim : int, default=2
        Dimensionality of the SPD matrices generated by the distributions.
    class_sep : float, default=1.0
        Parameter controlling the separability of the classes.
    class_disp : float, default=1.0
        Intra dispersion of the points sampled from each class.
    centers : ndarray, shape (2, n_dim, n_dim), default=None
        List with the centers of mass for each class. If None, the centers are
        sampled randomly based on class_sep.
    return_centers : bool, default=False
        If True, then return the centers of each cluster
    center_dataset : bool, default=False
        If True, re-center the simulated dataset to the Identity. If False,
        the dataset is centered around a random SPD matrix.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel. If -1 all CPUs are used.
    sampling_method : str, default='auto'
        Name of the sampling method used to sample samples_r. It can be
        'auto', 'slice' or 'rejection'. If it is 'auto', the sampling_method
        will be equal to 'slice' for n_dim != 2 and equal to
        'rejection' for n_dim = 2.

        .. versionadded:: 0.4

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
    .. versionadded:: 0.3

    """
    if not isinstance(class_sep, float):
        raise ValueError(f"class_sep must be a float (Got {class_sep})")

    rs = check_random_state(random_state)
    seeds = rs.randint(100, size=2)

    if centers is None:
        C0_in = np.eye(n_dim)  # first class mean at Identity at first
        Pv = rs.randn(n_dim, n_dim)  # create random tangent vector
        Pv = (Pv + Pv.T)/2   # symmetrize
        Pv = Pv / np.linalg.norm(Pv)  # normalize
        P = expm(Pv)  # take it back to the SPD manifold
        C1_in = powm(P, alpha=class_sep)  # control distance to Identity

    else:
        C0_in, C1_in = centers

    # sample data points from class 0
    X0 = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=C0_in,
        sigma=class_disp,
        random_state=seeds[0],
        n_jobs=n_jobs,
        sampling_method=sampling_method
    )
    y0 = np.zeros(n_matrices)

    # sample data points from class 1
    X1 = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=C1_in,
        sigma=class_disp,
        random_state=seeds[1],
        n_jobs=n_jobs,
        sampling_method=sampling_method
    )

    y1 = np.ones(n_matrices)

    # concatenate the samples
    X = np.concatenate([X0, X1])

    # re-center the dataset to the Identity
    M = mean_riemann(X)
    M_invsqrt = invsqrtm(M)
    X = M_invsqrt @ X @ M_invsqrt

    if not center_dataset:
        # center the dataset to a random SPD matrix
        M = generate_random_spd_matrix(n_dim=n_dim, random_state=rs)
        M_sqrt = sqrtm(M)
        X = M_sqrt @ X @ M_sqrt

    # concatenate the labels for each class
    y = np.concatenate([y0, y1]).astype(int)

    # randomly permute the samples of the dataset
    idx = rs.permutation(len(X))
    X, y = X[idx], y[idx]

    if return_centers:
        if centers is None:
            C0_out = mean_riemann(X[y == 0])
            C1_out = mean_riemann(X[y == 1])
        else:
            C0_out = C0_in
            C1_out = C1_in
        centers = np.stack([C0_out, C1_out])
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
    outlier_coeff: float, default=10
        Coefficient determining how to define an outlier data point, i.e. how
        many times the sigma parameter its distance to the mean should be.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    outliers : ndarray, shape (n_matrices, n_dim, n_dim)
        Set of simulated outlier matrix.

    Notes
    -----
    .. versionadded:: 0.3
    """

    n_dim = mean.shape[1]
    mean_sqrt = sqrtm(mean)

    outliers = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Oi = generate_random_spd_matrix(n_dim=n_dim, random_state=random_state)
        epsilon_num = outlier_coeff * sigma * n_dim
        epsilon_den = distance_riemann(Oi, np.eye(n_dim), squared=True)
        epsilon = np.sqrt(epsilon_num / epsilon_den)
        outliers[i] = mean_sqrt @ powm(Oi, epsilon) @ mean_sqrt

    return outliers


def make_classification_transfer(n_matrices, class_sep=3.0, class_disp=1.0,
                                 domain_sep=5.0, theta=0.0, stretch=1.0,
                                 random_state=None, class_names=[1, 2]):
    """Generate source and target toy datasets for transfer learning examples.

    Generate a dataset with 2x2 SPD matrices drawn from two Riemannian Gaussian
    distributions. The distributions have the same class dispersions and the
    distance between their centers of mass is an input parameter. We can
    stretch the target dataset and control a rotation matrix that maps the
    source to the target domains. This function is useful for testing
    classification or clustering methods on transfer learning applications.

    Parameters
    ----------
    n_matrices : int, default=100
        How many 2x2 matrices to generate for each class on each domain.
    class_sep : float, default=3.0
        Distance between the centers of the two classes.
    class_disp : float, default=1.0
        Dispersion of the data points to be sampled on each class.
    domain_sep : float, default=5.0
        Distance between the global means of each source and target datasets.
    theta : float, default=0.0
        Angle of the 2x2 rotation matrix from source to target dataset.
    stretch : float, default=1.0
        Factor to stretch the data points in target dataset. Note that when it
        is != 1.0 the class dispersions in target domain will be different than
        those in source domain (fixed at class_disp).
    random_state : None | int | RandomState instance, default=None
        Pass an int for reproducible output across multiple function calls.
    class_names : list, default=[1, 2]
        Names of classes.

    Returns
    -------
    X_enc : ndarray, shape (4*n_matrices, 2, 2)
        Set of SPD matrices.
    y_enc : ndarray, shape (4*n_matrices,)
        Extended labels for each data point.

    Notes
    -----
    .. versionadded:: 0.4
    """

    rs = check_random_state(random_state)
    seeds = rs.randint(100, size=4)

    # the examples considered here are always for 2x2 matrices
    n_dim = 2
    if len(class_names) != n_dim:
        raise ValueError("class_names must contain 2 elements")

    # create a source dataset with two classes and global mean at identity
    M1_source = np.eye(n_dim)  # first class mean at Identity at first
    X1_source = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=M1_source,
        sigma=class_disp,
        random_state=seeds[0])
    y1_source = [class_names[0]] * n_matrices
    Pv = rs.randn(n_dim, n_dim)  # create random tangent vector
    Pv = (Pv + Pv.T)/2  # symmetrize
    Pv /= np.linalg.norm(Pv)  # normalize
    P = expm(Pv)  # take it back to the SPD manifold
    M2_source = powm(P, alpha=class_sep)  # control distance to identity
    X2_source = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=M2_source,
        sigma=class_disp,
        random_state=seeds[1])
    y2_source = [class_names[1]] * n_matrices
    X_source = np.concatenate([X1_source, X2_source])
    M_source = mean_riemann(X_source)
    M_source_invsqrt = invsqrtm(M_source)
    # center the dataset to Identity
    X_source = M_source_invsqrt @ X_source @ M_source_invsqrt
    y_source = np.concatenate([y1_source, y2_source])

    # create target dataset based on the source dataset
    X1_target = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=M1_source,
        sigma=class_disp,
        random_state=seeds[2])
    X2_target = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=M2_source,
        sigma=class_disp,
        random_state=seeds[3])
    X_target = np.concatenate([X1_target, X2_target])
    M_target = mean_riemann(X_target)
    M_target_invsqrt = invsqrtm(M_target)
    # center the dataset to Identity
    X_target = M_target_invsqrt @ X_target @ M_target_invsqrt
    y_target = np.copy(y_source)

    # stretch the data points in target domain if needed
    if stretch != 1.0:
        X_target = powm(X_target, alpha=stretch)

    # move the points in X_target with a random matrix A = P * Q

    # create SPD matrix for the translation between domains
    Pv = rs.randn(n_dim, n_dim)  # create random tangent vector
    Pv = (Pv + Pv.T)/2  # symmetrize
    Pv /= np.linalg.norm(Pv)  # normalize
    P = expm(Pv)  # take it to the manifold
    P = powm(P, alpha=domain_sep)  # control distance to identity
    P = sqrtm(P)  # transport matrix

    # create orthogonal matrix for the rotation part
    Q = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    # transform the data points from the target domain
    A = P @ Q
    X_target = A @ X_target @ A.T

    # create array specifying the domain for each epoch
    domains = np.array(
        len(X_source)*['source_domain'] + len(X_target)*['target_domain']
    )

    # encode the labels and domains together
    X = np.concatenate([X_source, X_target])
    y = np.concatenate([y_source, y_target])
    X_enc, y_enc = encode_domains(X, y, domains)

    return X_enc, y_enc
