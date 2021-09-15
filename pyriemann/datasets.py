import numpy as np
from sklearn.utils.validation import check_random_state
from .sampling import generate_random_spd_matrix, sample_gaussian_spd


def make_gaussian_blobs(n_samples=100, n_dim=2, class_sep=1.0, class_disp=1.0,
                        random_state=42):
    """Generate SPD dataset with two classes sampled from Riemannian Gaussian

    Generate a dataset with SPD matrices generated from two Riemannian
    Gaussian distributions. The distributions have the same class dispersions
    and the distance between their centers of mass is an input parameter.

    Parameters
    ----------
    n_samples : int (default: 100)
        How many samples to generate for each class.
    n_dim : int (default: 2)
        Dimensionality of the SPD matrices generated by the distributions.
    class_sep : float (default: 1.0)
        Parameter controlling the separability of the classes.
    class_disp : float (default: 1.0)
        Intra dispersion of the points sampled from each class.
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    X : ndarray, shape (2*n_samples, n_dim, n_dim)
        ndarray of SPD matrices.
    y : ndarray shape (n_samples, 1)
        labels corresponding to each sample.

    Notes
    -----
    .. versionadded:: 0.2.8.dev

    """

    rs = check_random_state(random_state)

    # generate dataset for class 0
    CO = generate_random_spd_matrix(n_dim)
    X0 = sample_gaussian_spd(n_samples=n_samples,
                             mean=CO,
                             sigma=class_disp,
                             random_state=random_state)
    y0 = np.zeros(n_samples)

    # generate dataset for class 1
    epsilon = np.exp(class_sep/np.sqrt(n_dim))
    C1 = epsilon * CO
    X1 = sample_gaussian_spd(n_samples=n_samples,
                             mean=C1,
                             sigma=class_disp,
                             random_state=random_state)
    y1 = np.ones(n_samples)

    X = np.concatenate([X0, X1])
    y = np.concatenate([y0, y1])
    idx = rs.permutation(len(X))
    X = X[idx]
    y = y[idx]

    return X, y