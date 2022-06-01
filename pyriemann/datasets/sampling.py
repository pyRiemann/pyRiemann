from functools import partial
import warnings
import numpy as np
from sklearn.utils import check_random_state

from ..utils.base import sqrtm, expm
from ..utils.test import is_sym_pos_semi_def as is_spsd


def _pdf_r(r, sigma):
    """Pdf for the log of eigenvalues of a SPD matrix

    Probability density function for the logarithm of the eigenvalues of a SPD
    matrix samples from the Riemannian Gaussian distribution. See Said et al.
    "Riemannian Gaussian distributions on the space of symmetric positive
    definite matrices" (2017) for the mathematical details.

    Parameters
    ----------
    r : ndarray, shape (n_dim,)
        Vector with the logarithm of the eigenvalues of a SPD matrix.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.

    Returns
    -------
    p : float
        Probability density function applied to data point r
    """

    if (sigma <= 0):
        raise ValueError(f'sigma must be a positive number (Got {sigma})')

    n_dim = len(r)
    partial_1 = -np.sum(r**2) / sigma**2
    partial_2 = 0
    for i in range(n_dim):
        for j in range(i + 1, n_dim):
            partial_2 = partial_2 + np.log(np.sinh(np.abs(r[i] - r[j]) / 2))

    return np.exp(partial_1 + partial_2)


def _slice_sampling(ptarget, n_samples, x0, n_burnin=20, thin=10,
                    random_state=None):
    """Slice sampling procedure

    Implementation of a slice sampling algorithm for sampling from any target
    pdf or a multiple of it. The implementation follows the description given
    in page 375 of David McKay's book "Information Theory, Inference, and
    Learning Algorithms" (2003).

    Parameters
    ----------
    ptarget : function with one input
        The target pdf to sample from or a multiple of it.
    n_samples : int
        How many samples to get from the ptarget distribution.
    x0 : array
        Initial state for the MCMC procedure. Note that the shape of this array
        defines the dimensionality n_dim of the data points to be sampled.
    n_burnin : int
        How many samples to discard from the beginning of the chain generated
        by the slice sampling procedure. Usually the first samples are prone to
        non-stationary behavior and do not follow very well the target pdf.
    thin : int
        Thinning factor for the slice sampling procedure. MCMC samples are
        often correlated between them, so taking one sample every `thin`
        samples can help reducing this correlation. Note that this makes the
        algorithm actually sample `thin x n_samples` samples from the pdf, so
        expect the whole sampling procedure to take longer
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    samples : ndarray, shape (n_samples, n_dim)
        Samples from the target pdf.
    """

    if (n_samples <= 0) or (not isinstance(n_samples, int)):
        raise ValueError(
            f'n_samples must be a positive integer (Got {n_samples})')
    if (n_burnin <= 0) or (not isinstance(n_burnin, int)):
        raise ValueError(
            f'n_samples must be a positive integer (Got {n_burnin})')
    if (thin <= 0) or (not isinstance(thin, int)):
        raise ValueError(f'thin must be a positive integer (Got {thin})')

    rs = check_random_state(random_state)
    w = 1.0  # initial bracket width
    xt = np.copy(x0)

    n_dim = len(x0)
    samples = []
    n_samples_total = (n_samples + n_burnin) * thin

    for _ in range(n_samples_total):

        for i in range(n_dim):

            ei = np.zeros(n_dim)
            ei[i] = 1

            # step 1 : evaluate ptarget(xt)
            Px = ptarget(xt)

            # step 2 : draw vertical coordinate uprime ~ U(0, ptarget(xt))
            uprime_i = Px * rs.rand()

            # step 3 : create a horizontal interval (xl_i, xr_i) enclosing xt_i
            r = rs.rand()
            xl_i = xt[i] - r * w
            xr_i = xt[i] + (1-r) * w
            while ptarget(xt + (xl_i - xt[i]) * ei) > uprime_i:
                xl_i = xl_i - w
            while ptarget(xt + (xr_i - xt[i]) * ei) > uprime_i:
                xr_i = xr_i + w

            # step 4 : loop
            while True:
                xprime_i = xl_i + (xr_i - xl_i) * rs.rand()
                Px = ptarget(xt + (xprime_i - xt[i]) * ei)
                if Px > uprime_i:
                    break
                else:
                    if xprime_i > xt[i]:
                        xr_i = xprime_i
                    else:
                        xl_i = xprime_i

            # store coordinate i of new sample
            xt = np.copy(xt)
            xt[i] = xprime_i

        samples.append(xt)

    samples = np.array(samples)[(n_burnin * thin):][::thin]

    return samples


def _sample_parameter_r(n_samples, n_dim, sigma, random_state=None):
    """Sample the r parameters of a Riemannian Gaussian distribution

    Sample the logarithm of the eigenvalues of a SPD matrix following a
    Riemannian Gaussian distribution.

    See https://arxiv.org/pdf/1507.01760.pdf for the mathematical details

    Parameters
    ----------
    n_samples : int
        How many samples to generate.
    n_dim : int
        Dimensionality of the SPD matrices to be sampled.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    r_samples : ndarray, shape (n_samples, n_dim)
        Samples of the r parameters of the Riemannian Gaussian distribution.
    """

    rs = check_random_state(random_state)
    x0 = rs.randn(n_dim)
    ptarget = partial(_pdf_r, sigma=sigma)
    r_samples = _slice_sampling(
        ptarget, n_samples=n_samples, x0=x0, random_state=random_state)

    return r_samples


def _sample_parameter_U(n_samples, n_dim, random_state=None):
    """Sample the U parameters of a Riemannian Gaussian distribution

    Sample the eigenvectors of a SPD matrix following a Riemannian Gaussian
    distribution.

    See https://arxiv.org/pdf/1507.01760.pdf for the mathematical details

    Parameters
    ----------
    n_samples : int
        How many samples to generate.
    n_dim : int
        Dimensionality of the SPD matrices to be sampled.
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    u_samples : ndarray, shape (n_samples, n_dim)
        Samples of the U parameters of the Riemannian Gaussian distribution
    """

    rs = check_random_state(random_state)
    u_samples = np.zeros((n_samples, n_dim, n_dim))
    for i in range(n_samples):
        A = rs.randn(n_dim, n_dim)
        Q, _ = np.linalg.qr(A)
        u_samples[i] = Q

    return u_samples


def _sample_gaussian_spd_centered(n_matrices, n_dim, sigma, random_state=None):
    """Sample a Riemannian Gaussian distribution centered at the Identity

    Sample SPD matrices from a Riemannian Gaussian distribution centered at the
    Identity, which has the role of the origin in the SPD manifold, and
    dispersion parametrized by sigma. See [1]_ for the mathematical details.

    Parameters
    ----------
    n_matrices : int
        How many matrices to generate.
    n_dim : int
        Dimensionality of the SPD matrices to be sampled.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    samples : ndarray, shape (n_matrices, n_dim, n_dim)
        Samples of the Riemannian Gaussian distribution

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] S. Said, L. Bombrun, Y. Berthoumieu, and J. Manton, “Riemannian
        Gaussian distributions on the space of symmetric positive definite
        matrices”, IEEE Trans Inf Theory, vol. 63, pp. 2153–2170, 2017.
        https://arxiv.org/pdf/1507.01760.pdf
    """

    samples_r = _sample_parameter_r(n_samples=n_matrices,
                                    n_dim=n_dim,
                                    sigma=sigma,
                                    random_state=random_state)
    samples_U = _sample_parameter_U(n_samples=n_matrices,
                                    n_dim=n_dim,
                                    random_state=random_state)

    samples = np.zeros((n_matrices, n_dim, n_dim))
    for i in range(n_matrices):
        Ui = samples_U[i]
        ri = samples_r[i]
        samples[i] = Ui.T @ np.diag(np.exp(ri)) @ Ui
        samples[i] = 0.5 * (samples[i] + samples[i].T)

    return samples


def sample_gaussian_spd(n_matrices, mean, sigma, random_state=None):
    """Sample a Riemannian Gaussian distribution

    Sample SPD matrices from a Riemannian Gaussian distribution centered at
    mean and with dispersion parametrized by sigma. This distribution has been
    defined in [1]_ and generalizes the notion of a Gaussian distribution to
    the space of SPD matrices. The sampling is based on a spectral
    factorization of SPD matrices in terms of their eigenvectors (U-parameters)
    and the log of the eigenvalues (r-parameters).

    Parameters
    ----------
    n_matrices : int
        How many matrices to generate.
    mean : ndarray, shape (n_dim, n_dim)
        Center of the Riemannian Gaussian distribution.
    sigma : float
        Dispersion of the Riemannian Gaussian distribution.
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    samples : ndarray, shape (n_matrices, n_dim, n_dim)
        Samples of the Riemannian Gaussian distribution

    Notes
    -----
    .. versionadded:: 0.2.8

    References
    ----------
    .. [1] S. Said, L. Bombrun, Y. Berthoumieu, and J. Manton, “Riemannian
        Gaussian distributions on the space of symmetric positive definite
        matrices”, IEEE Trans Inf Theory, vol. 63, pp. 2153–2170, 2017.
        https://arxiv.org/pdf/1507.01760.pdf
    """

    n_dim = mean.shape[0]
    samples_centered = _sample_gaussian_spd_centered(
        n_matrices=n_matrices,
        n_dim=n_dim,
        sigma=sigma,
        random_state=random_state
    )

    # apply the parallel transport to mean on each of the samples
    mean_sqrt = sqrtm(mean)
    samples = mean_sqrt @ samples_centered @ mean_sqrt

    if not is_spsd(samples):
        msg = "Some of the sampled matrices are very badly conditioned and \
               may not behave numerically as a SPD matrix. Try sampling again \
               or reducing the dimensionality of the matrix."
        warnings.warn(msg)

    return samples


def generate_random_spd_matrix(n_dim, random_state=None, *, mat_mean=.0,
                               mat_std=1.):
    """Generate a random SPD matrix

    Parameters
    ----------
    n_dim : int
        Dimensionality of the matrix to sample.
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.
    mat_mean : float (default 0.0)
        Mean of random values to generate matrix.
    mat_std : float (default 1.0)
        Standard deviation of random values to generate matrix.

    Returns
    -------
    C : ndarray, shape (n_dim, n_dim)
        Random SPD matrix

    Notes
    -----
    .. versionadded:: 0.2.8
    """

    if (n_dim <= 0) or (not isinstance(n_dim, int)):
        raise ValueError(
            f'n_samples must be a positive integer (Got {n_dim})')

    rs = check_random_state(random_state)
    A = mat_mean + mat_std * rs.randn(n_dim, n_dim)
    A = 0.5 * (A + A.T)
    C = expm(A)

    if not is_spsd(C):
        msg = "The sampled matrix is very badly conditioned and may not \
               behave numerically as a SPD matrix. Try sampling again or \
               reducing the dimensionality of the matrix."
        warnings.warn(msg)

    return C
