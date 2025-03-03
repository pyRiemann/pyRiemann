import numpy as np
from sklearn.utils.validation import check_random_state

from .sampling import sample_gaussian_spd
from ..transfer import encode_domains
from ..utils.mean import mean_riemann
from ..utils.distance import distance_riemann
from ..utils.base import ctranspose, invsqrtm, powm, sqrtm, expm


mat_kinds = ["real", "sym", "comp", "spd", "spsd", "herm", "hpd", "hpsd"]


def make_matrices(n_matrices, n_dim, kind, rs=None, return_params=False,
                  evals_low=0.5, evals_high=2.0, eigvecs_same=False,
                  eigvecs_mean=0.0, eigvecs_std=1.0):
    """Generate square matrices, with specific properties.

    Parameters
    ----------
    n_matrices : int
        Number of square matrices to generate.
    n_dim : int
        Dimension of square matrices to generate.
    kind : {"real", "sym", "spd", "spsd", "comp", "herm", "hpd", "hpsd"}
        Kind of square matrices to generate:

        - "real" for real-valued matrices;
        - "sym" for symmetric real-valued matrices;
        - "spd" for symmetric positive-definite matrices;
        - "spsd" for symmetric positive semi-definite matrices;
        - "comp" for complex-valued matrices;
        - "herm" for Hermitian matrices;
        - "hpd" for Hermitian positive-definite matrices;
        - "hpsd" for Hermitian positive semi-definite matrices.
    rs : int | RandomState instance | None, default=None
        Random state for reproducible output across multiple function calls.
    return_params : bool, default=False
        If True, returns evals and evecs for "spd", "spsd", "hpd" and "hpsd".
    evals_low : float, default=0.5
        Lowest value of the uniform distribution to draw eigen values.
    evals_high : float, default=2.0
        Highest value of the uniform distribution to draw eigen values.
    eigvecs_same : bool, default=False
        If True, uses the same eigen vectors for all matrices.
    eigvecs_mean : float, default=0.0
        Mean of the normal distribution to draw eigen vectors.

        .. versionadded:: 0.8
    eigvecs_std : float, default=1.0
        Standard deviation of the normal distribution to draw eigen vectors.

        .. versionadded:: 0.8

    Returns
    -------
    mats : ndarray, shape (n_matrices, n_dim, n_dim)
        Set of generated square matrices.
    evals : ndarray, shape (n_matrices, n_dim)
        Eigen values used for "spd", "spsd", "hpd" and "hpsd".
        Only returned if ``return_params=True``.
    evecs : ndarray, shape (n_matrices, n_dim, n_dim) or (n_dim, n_dim)
        Eigen vectors used for "spd", "spsd", "hpd" and "hpsd".
        Only returned if ``return_params=True``.

    Notes
    -----
    .. versionadded:: 0.5
    .. versionchanged:: 0.8
        Add support for kinds "sym" and "herm".
    """
    if kind not in mat_kinds:
        raise ValueError(f"Unsupported matrix kind: {kind}")
    rs = check_random_state(rs)

    X = eigvecs_std * rs.randn(n_matrices, n_dim, n_dim) + eigvecs_mean

    if kind == "real":
        return X
    if kind == "sym":
        return X + X.transpose(0, 2, 1)
    if kind in ["comp", "herm", "hpd", "hpsd"]:
        Y = eigvecs_std * rs.randn(n_matrices, n_dim, n_dim) + eigvecs_mean
        if kind == "herm":
            return X + X.transpose(0, 2, 1) + 1j * (Y - Y.transpose(0, 2, 1))
        X = X + 1j * Y
        if kind == "comp":
            return X

    # eigen values
    if evals_low <= 0.0:
        raise ValueError(
            f"Lowest value must be strictly positive (Got {evals_low})."
        )
    if evals_high <= evals_low:
        raise ValueError(
            "Highest value must be superior to lowest value "
            f"(Got {evals_high} and {evals_low})."
        )
    evals = rs.uniform(evals_low, evals_high, size=(n_matrices, n_dim))
    if kind in ("spsd", "hpsd"):
        evals[..., -1] = 1e-10  # last eigen value set to almost zero

    # eigen vectors
    if eigvecs_same:
        X = X[0]
    if np.__version__ < "1.22.0" and X.ndim > 2:
        evecs = np.array([np.linalg.qr(x)[0] for x in X])
    else:
        evecs = np.linalg.qr(X)[0]

    # conjugation
    if eigvecs_same:
        mats = np.empty((n_matrices, n_dim, n_dim), dtype=X.dtype)
        for i in range(n_matrices):
            mats[i] = (evecs * evals[i]) @ evecs.conj().T
    else:
        mats = (evecs * evals[:, np.newaxis, :]) @ ctranspose(evecs)

    if return_params:
        return mats, evals, evecs
    else:
        return mats


def make_masks(n_masks, n_dim0, n_dim1_min, rs=None):
    """Generate masks defined as semi-orthogonal matrices.

    Parameters
    ----------
    n_masks : int
        Number of masks to generate.
    n_dim0 : int
        First dimension of masks.
    n_dim1_min : int
        Minimal value for second dimension of masks.
    rs : int | RandomState instance | None, default=None
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
                        sampling_method="auto"):
    """Generate SPD matrices for two classes.

    Generate a set of SPD matrices drawn from Riemannian Gaussian
    distributions, one per class. Currently, it supports two classes.
    The distributions have the same dispersions.
    Useful for testing classification or clustering methods.

    Parameters
    ----------
    n_matrices : int, default=100
        Number of matrices to generate for each class.
    n_dim : int, default=2
        Dimensionality of the generated SPD matrices.
    class_sep : float, default=1.0
        Distance between the centers of the classes.
    class_disp : float, default=1.0
        Dispersion of the matrices for each class.
    centers : None | ndarray, shape (2, n_dim, n_dim), default=None
        Centers for each class.
        If None, the centers are drawn randomly based on class_sep.
    return_centers : bool, default=False
        If True, return the centers of each class.
    center_dataset : bool, default=False
        If True, re-center dataset to the Identity.
        If False, dataset is centered around a random SPD matrix.
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel. If -1 all CPUs are used.
    sampling_method : {"auto", "slice", "rejection"}, default="auto"
        Method used to sample eigenvalues: "auto", "slice" or "rejection".
        If "auto", sampling_method will be equal to "slice" for n_dim != 2 and
        equal to "rejection" for n_dim = 2.

        .. versionadded:: 0.4

    Returns
    -------
    X : ndarray, shape (2*n_matrices, n_dim, n_dim)
        Set of SPD matrices, for two classes.
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

    # sample matrices from class 0
    X0 = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=C0_in,
        sigma=class_disp,
        random_state=seeds[0],
        n_jobs=n_jobs,
        sampling_method=sampling_method
    )
    y0 = np.zeros(n_matrices)

    # sample matrices from class 1
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
        M = make_matrices(n_matrices=1, n_dim=n_dim, kind="spd", rs=rs)[0]
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
    """Generate outlier matrices.

    Generate matrices that are outliers for a given Riemannian Gaussian
    distribution with fixed mean and dispersion.

    Parameters
    ----------
    n_matrices : int
        Number of matrices to generate.
    mean : ndarray, shape (n_dim, n_dim)
        Center of the Riemannian Gaussian distribution.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    outlier_coeff: float, default=10
        Coefficient determining how to define an outlier, i.e. how
        many times the sigma parameter its distance to the mean should be.
    random_state : int | RandomState instance | None, default=None
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    outliers : ndarray, shape (n_matrices, n_dim, n_dim)
        Set of generated outlier matrices.

    Notes
    -----
    .. versionadded:: 0.3
    """

    n_dim = mean.shape[1]
    mean_sqrt = sqrtm(mean)

    outliers = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Oi = make_matrices(1, n_dim=n_dim, kind="spd", rs=random_state)[0]
        epsilon_num = outlier_coeff * sigma * n_dim
        epsilon_den = distance_riemann(Oi, np.eye(n_dim), squared=True)
        epsilon = np.sqrt(epsilon_num / epsilon_den)
        outliers[i] = mean_sqrt @ powm(Oi, epsilon) @ mean_sqrt

    return outliers


def make_classification_transfer(
    n_matrices,
    class_sep=3.0,
    class_disp=1.0,
    domain_sep=5.0,
    theta=0.0,
    stretch=1.0,
    random_state=None,
    class_names=[1, 2],
    domain_names=["source_domain", "target_domain"],
):
    """Generate 2x2 SPD matrices for two classes in source and target domains.

    Generate a set of 2x2 SPD matrices drawn from Riemannian Gaussian
    distributions, one per class and per domain.
    Currently, it supports two classes and two domains.
    The distributions have the same dispersions.
    You can stretch the target domain and control a rotation matrix that maps
    the source domain to the target domain.
    Useful for testing classification or clustering methods on transfer
    learning applications.

    Parameters
    ----------
    n_matrices : int
        Number of 2x2 matrices to generate for each class on each domain.
    class_sep : float, default=3.0
        Distance between the centers of the classes.
    class_disp : float, default=1.0
        Dispersion of the matrices for each class.
    domain_sep : float, default=5.0
        Distance between the global means of each source and target domains.
    theta : float, default=0.0
        Angle of the 2x2 rotation matrix from source to target domain.
    stretch : float, default=1.0
        Factor to stretch the matrices in target domain. Note that when it
        is != 1.0 the class dispersions in target domain will be different than
        those in source domain (fixed at class_disp).
    random_state : None | int | RandomState instance, default=None
        Pass an int for reproducible output across multiple function calls.
    class_names : list, default=[1, 2]
        Names of classes.
    domain_names : list, default=["source_domain", "target_domain"]
        Names of domains, source and target.

        .. versionadded:: 0.8

    Returns
    -------
    X_enc : ndarray, shape (4*n_matrices, 2, 2)
        Set of 2x2 SPD matrices, for two classes and two domains.
    y_enc : ndarray, shape (4*n_matrices,)
        Extended labels for each matrix.

    Notes
    -----
    .. versionadded:: 0.4
    """

    rs = check_random_state(random_state)
    seeds = rs.randint(100, size=4)

    n_dim = 2
    if len(class_names) != 2:
        raise ValueError("class_names must contain 2 elements")
    if len(domain_names) != 2:
        raise ValueError("domain_names must contain 2 elements")

    # create a source domain with two classes and global mean at identity
    M1_source = np.eye(n_dim)  # first class mean at Identity at first
    X1_source = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=M1_source,
        sigma=class_disp,
        random_state=seeds[0],
    )
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
        random_state=seeds[1],
    )
    y2_source = [class_names[1]] * n_matrices
    X_source = np.concatenate([X1_source, X2_source])
    M_source = mean_riemann(X_source)
    M_source_invsqrt = invsqrtm(M_source)
    # center the domain to Identity
    X_source = M_source_invsqrt @ X_source @ M_source_invsqrt
    y_source = np.concatenate([y1_source, y2_source])

    # create target domain based on the source domain
    X1_target = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=M1_source,
        sigma=class_disp,
        random_state=seeds[2],
    )
    X2_target = sample_gaussian_spd(
        n_matrices=n_matrices,
        mean=M2_source,
        sigma=class_disp,
        random_state=seeds[3],
    )
    X_target = np.concatenate([X1_target, X2_target])
    M_target = mean_riemann(X_target)
    M_target_invsqrt = invsqrtm(M_target)
    # center the domain to Identity
    X_target = M_target_invsqrt @ X_target @ M_target_invsqrt
    y_target = np.copy(y_source)

    # stretch the matrices in target domain if needed
    if stretch != 1.0:
        X_target = powm(X_target, alpha=stretch)

    # move the matrices in X_target with a random matrix A = P * Q

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

    # transform the matrices from the target domain
    A = P @ Q
    X_target = A @ X_target @ A.T

    # create array specifying the domain for each matrix
    domains = np.array(
        len(X_source) * [domain_names[0]] + len(X_target) * [domain_names[1]]
    )

    # encode the labels and domains together
    X = np.concatenate([X_source, X_target])
    y = np.concatenate([y_source, y_target])
    X_enc, y_enc = encode_domains(X, y, domains)

    return X_enc, y_enc
