import numpy as np
from sklearn.utils.validation import check_random_state

from ..geometry.base import ctranspose, invsqrtm, powm, sqrtm, expm
from ..geometry.distance import distance_riemann
from ..geometry.mean import mean_riemann
from ..transfer import encode_domains
from .sampling import sample_gaussian


def _make_eyes(n_matrices, n_dim):
    """Generate a 3d array of stacked np.eye matrices.

    Parameters
    ----------
    n_matrices : int
        Number of matrices to generate.
    n_dim : int
        Dimension of eye matrices to generate.

    Returns
    -------
    X : ndarray, shape (n_matrices, n_dim, n_dim)
        Set of np.eye matrices.

    Notes
    -----
    .. versionadded:: 0.10
    """
    return np.repeat(np.eye(n_dim)[np.newaxis, :, :], n_matrices, axis=0)


mat_kinds = [
    "real",
    "comp",
    "inv",
    "orth",
    "sym",
    "cinv",
    "unit",
    "spd",
    "spsd",
    "herm",
    "hpd",
    "hpsd",
]


def make_matrices(n_matrices, n_dim, kind, rs=None, return_params=False,
                  evals_low=0.5, evals_high=2.0, eigvecs_same=False,
                  eigvecs_mean=0.0, eigvecs_std=1.0):
    """Generate matrices with specific properties.

    Parameters
    ----------
    n_matrices : int
        Number of matrices to generate.
    n_dim : int | list of int
        If int, dimension of square matrices to generate.
        If list, dimensions of "real" or "comp" matrices to generate.

        .. versionchanged:: 0.10
            Add support for list of ints.
    kind : {"real", "comp", "inv", "orth, "sym", "spd", "spsd", "cinv", \
            "unit", "herm", "hpd", "hpsd"}
        Kind of matrices to generate:

        - "real" for real-valued matrices;
        - "comp" for complex-valued matrices.

        Kind of square matrices to generate:

        - "inv" for invertible real-valued matrices;
        - "orth" for orthogonal matrices;
        - "sym" for symmetric real-valued matrices;
        - "spd" for symmetric positive-definite matrices;
        - "spsd" for symmetric positive semi-definite matrices;
        - "cinv" for invertible complex-valued matrices;
        - "unit" for unitary matrices;
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
    mats : ndarray, shape (n_matrices, n_dim, n_dim) or (n_matrices, \\*n_dim)
        Set of generated matrices.
    evals : ndarray, shape (n_matrices, n_dim)
        Eigen values used for "spd", "spsd", "hpd" and "hpsd".
        Only returned if ``return_params=True``.
    evecs : ndarray, shape (n_matrices, n_dim, n_dim) or (n_dim, n_dim)
        Eigen vectors used for "spd", "spsd", "hpd" and "hpsd".
        Only returned if ``return_params=True``.

    Notes
    -----
    .. versionadded:: 0.3
    .. versionchanged:: 0.5
        Rename ``make_covariances`` into ``make_matrices``.
    .. versionchanged:: 0.8
        Add ``"sym"`` and ``"herm"`` options to parameter ``kind``.
    .. versionchanged:: 0.10
        Add options to parameter ``kind``: non-square matrices ``"real"`` and
        ``"comp"``; ``"inv"``, ``"cinv"``, ``"orth"`` and ``"unit"`` .
    """
    rs = check_random_state(rs)

    if isinstance(n_dim, list):
        if kind not in ["real", "comp"]:
            raise ValueError(f"Unsupported matrix kind: {kind}")
        X = rs.randn(n_matrices, *n_dim)
        if kind == "comp":
            X = X + 1j * rs.randn(n_matrices, *n_dim)
        return X

    if not isinstance(n_dim, int):
        raise ValueError(f"Unsupported n_dim type: {type(n_dim)}")
    if kind not in mat_kinds:
        raise ValueError(f"Unsupported matrix kind: {kind}")

    if kind in ["inv", "cinv"]:
        while True:
            X = rs.randn(n_matrices, n_dim, n_dim)
            if kind == "cinv":
                X = X + 1j * rs.randn(n_matrices, n_dim, n_dim)
            if np.all(np.linalg.det(X) != 0):
                return X

    X = eigvecs_std * rs.randn(n_matrices, n_dim, n_dim) + eigvecs_mean

    if kind == "unit":
        Y = eigvecs_std * rs.randn(n_matrices, n_dim, n_dim) + eigvecs_mean
        X = X + 1j * Y
    if kind in ["orth", "unit"]:
        return np.linalg.qr(X)[0]

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
    evecs = np.linalg.qr(X)[0]

    # conjugation
    if eigvecs_same:
        mats = np.empty((n_matrices, n_dim, n_dim), dtype=X.dtype)
        for i in range(n_matrices):
            mats[i] = (evecs * evals[i]) @ ctranspose(evecs)
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

        .. versionadded:: 0.4
    return_centers : bool, default=False
        If True, return the centers of each class.
    center_dataset : bool, default=False
        If True, re-center dataset to the Identity.
        If False, dataset is centered around a random SPD matrix.

        .. versionadded:: 0.4
    random_state : int, RandomState instance or None, default=None
        Pass an int for reproducible output across multiple function calls.
    n_jobs : int, default=1
        The number of jobs to use for the computation. This works by computing
        each of the class centroid in parallel. If -1 all CPUs are used.

        .. versionadded:: 0.3
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
    .. versionchanged:: 0.3
        Add parameter ``n_jobs``.
    .. versionchanged:: 0.4
        Add parameters ``centers``, ``center_dataset`` and ``sampling_method``.
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
    X0 = sample_gaussian(
        n_matrices=n_matrices,
        mean=C0_in,
        sigma=class_disp,
        random_state=seeds[0],
        n_jobs=n_jobs,
        sampling_method=sampling_method
    )
    y0 = np.zeros(n_matrices)

    # sample matrices from class 1
    X1 = sample_gaussian(
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


def _sample_tangent_vector(rs, n_dim, is_complex):
    """Draw a random unit-norm tangent vector at the identity."""
    Pv = rs.randn(n_dim, n_dim)
    if is_complex:
        Pv = Pv + 1j * rs.randn(n_dim, n_dim)
    Pv = (Pv + ctranspose(Pv)) / 2  # symmetrize
    Pv /= np.linalg.norm(Pv)  # normalize
    return Pv


def _make_rotation(theta, n_dim):
    """Rotation of angle theta in the plane of the two first axes."""
    Q = np.eye(n_dim)
    Q[0, 0], Q[0, 1] = np.cos(theta), -np.sin(theta)
    Q[1, 0], Q[1, 1] = np.sin(theta), np.cos(theta)
    return Q


def _check_target_param(param, n_targets, name):
    """Check a parameter defined for a scalar or for each target domain."""
    param = np.atleast_1d(param)
    if param.ndim != 1:
        raise ValueError(f"{name} must be a scalar or a 1d array")
    if param.size == 1:
        return np.repeat(param, n_targets)
    if param.size != n_targets:
        raise ValueError(
            f"{name} must be a scalar, or contain {n_targets} elements, one "
            f"for each target domain (Got {param.size})"
        )
    return param


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
    n_dim=2,
    kind="spd",
):
    """Generate SPD or HPD matrices for several classes and domains.

    Generate a set of SPD or HPD matrices drawn from Riemannian Gaussian
    distributions, one per class and per domain.
    The distributions have the same dispersions.
    The first domain is the source domain, and its global mean is the identity
    matrix. Each other domain is a target domain, obtained by stretching the
    matrices and by applying a transformation controlling its distance and its
    rotation with respect to the source domain.
    Useful for testing classification or clustering methods on transfer
    learning applications.

    Parameters
    ----------
    n_matrices : int
        Number of matrices to generate for each class on each domain.
    class_sep : float, default=3.0
        Distance between the center of the first class and the centers of the
        other classes.
    class_disp : float, default=1.0
        Dispersion of the matrices for each class.
    domain_sep : float | array-like, default=5.0
        Distance between the global means of the source domain and of each
        target domain. If a scalar, the same distance is used for all target
        domains.
    theta : float | array-like, default=0.0
        Angle of the rotation matrix from source domain to each target domain,
        in the plane spanned by the two first axes. If a scalar, the same
        angle is used for all target domains.
    stretch : float | array-like, default=1.0
        Factor to stretch the matrices in each target domain. Note that when it
        is != 1.0 the class dispersions in target domain will be different than
        those in source domain (fixed at class_disp). If a scalar, the same
        factor is used for all target domains.
    random_state : None | int | RandomState instance, default=None
        Pass an int for reproducible output across multiple function calls.
    class_names : list, default=[1, 2]
        Names of classes, at least two.
    domain_names : list, default=["source_domain", "target_domain"]
        Names of domains, at least two. The first one is the source domain,
        the other ones are target domains.

        .. versionadded:: 0.8
    n_dim : int, default=2
        Dimension of the generated matrices, at least two.

        .. versionadded:: 0.13
    kind : {"spd", "hpd"}, default="spd"
        Kind of matrices to generate: symmetric positive-definite, or Hermitian
        positive-definite.

        .. versionadded:: 0.13

    Returns
    -------
    X_enc : ndarray, shape (n_matrices_tot, n_dim, n_dim)
        Set of SPD or HPD matrices, where n_matrices_tot is equal to
        n_matrices x len(class_names) x len(domain_names).
    y_enc : ndarray, shape (n_matrices_tot,)
        Extended labels for each matrix.

    Notes
    -----
    .. versionadded:: 0.4
    .. versionchanged:: 0.8
        Add parameter ``domain_names``.
    .. versionchanged:: 0.13
        Add support for more than two classes, for more than two domains, for
        matrices of dimension higher than two, and for HPD matrices.
        Parameters ``domain_sep``, ``theta`` and ``stretch`` can be defined for
        each target domain.
    """

    n_classes, n_domains = len(class_names), len(domain_names)
    if n_classes < 2:
        raise ValueError(
            f"class_names must contain at least 2 elements (Got {n_classes})"
        )
    if n_domains < 2:
        raise ValueError(
            f"domain_names must contain at least 2 elements (Got {n_domains})"
        )
    if not isinstance(n_dim, (int, np.integer)) or n_dim < 2:
        raise ValueError(
            f"n_dim must be an integer at least equal to 2 (Got {n_dim})"
        )
    if kind not in ("spd", "hpd"):
        raise ValueError(f"Unsupported matrix kind: {kind}")

    n_targets = n_domains - 1
    domain_seps = _check_target_param(domain_sep, n_targets, "domain_sep")
    thetas = _check_target_param(theta, n_targets, "theta")
    stretches = _check_target_param(stretch, n_targets, "stretch")

    is_complex = kind == "hpd"
    rs = check_random_state(random_state)
    seeds = rs.randint(100, size=n_classes * n_domains)

    # create the class means, the first one at identity
    means = [np.eye(n_dim, dtype=complex if is_complex else float)]
    for _ in range(n_classes - 1):
        Pv = _sample_tangent_vector(rs, n_dim, is_complex)
        P = expm(Pv)  # take it back to the manifold
        means.append(powm(P, alpha=class_sep))  # control distance to identity

    # create the transformations from source domain to each target domain
    transfos = []
    for i in range(n_targets):
        # create SPD/HPD matrix for the translation between domains
        Pv = _sample_tangent_vector(rs, n_dim, is_complex)
        P = expm(Pv)  # take it to the manifold
        P = powm(P, alpha=domain_seps[i])  # control distance to identity
        P = sqrtm(P)  # transport matrix
        # create orthogonal matrix for the rotation part
        Q = _make_rotation(thetas[i], n_dim)
        transfos.append(P @ Q)

    X, y, domains = [], [], []
    for d in range(n_domains):
        X_d = np.concatenate([
            sample_gaussian(
                n_matrices=n_matrices,
                mean=means[k],
                sigma=class_disp,
                random_state=seeds[d * n_classes + k],
            )
            for k in range(n_classes)
        ])
        # center the domain to identity
        M_invsqrt = invsqrtm(mean_riemann(X_d))
        X_d = M_invsqrt @ X_d @ M_invsqrt

        if d > 0:
            # stretch the matrices in target domain if needed
            if stretches[d - 1] != 1.0:
                X_d = powm(X_d, alpha=stretches[d - 1])
            # move the matrices with a matrix A = P * Q
            A = transfos[d - 1]
            X_d = A @ X_d @ ctranspose(A)

        X.append(X_d)
        y.append(np.repeat(class_names, n_matrices))
        domains.append(np.repeat(domain_names[d], len(X_d)))

    # encode the labels and domains together
    X_enc, y_enc = encode_domains(
        np.concatenate(X), np.concatenate(y), np.concatenate(domains)
    )

    return X_enc, y_enc
