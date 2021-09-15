import numpy as np
from functools import partial
from pyriemann.utils.base import sqrtm
from sklearn.utils import check_random_state


def _pdf_r(r, sigma):
    """pdf for the log of eigenvalues of a SPD matrix

    Probability density function for the logarithm of the eigenvalues of a SPD
    matrix samples from the Riemannian Gaussian distribution [1]_.

    Parameters
    ----------
    r : ndarray shape (n_dim,)
        vector defines in R^n_dim
    sigma : float
        dispersion of the Riemannian Gaussian distribution

    Returns
    -------
    p : float
        pdf applied to data point r

    Notes
    -----
    .. versionadded:: 0.2.8.dev

    References
    ----------
    .. [1] S. Said, L. Bombrun, Y. Berthoumieu, and J. Manton, “Riemannian
        Gaussian distributions on the space of symmetric positive definite
        matrices”, IEEE Trans Inf Theory, vol. 63, pp. 2153–2170, 2017.
        https://arxiv.org/pdf/1507.01760.pdf
    """

    n_dim = len(r)
    partial_1 = -np.sum(r**2)/sigma**2
    partial_2 = 0
    for i in range(n_dim):
        for j in range(i+1, n_dim):
            partial_2 = partial_2 + np.log(np.sinh(np.abs(r[i]-r[j])/2))

    return np.exp(partial_1 + partial_2)


def _slice_sampling(ptarget, n_samples, x0, n_burnin=20, thin=10,
                    random_state=42):
    """Slice sampling procedure

    Implementation of a slice sampling algorithm for sampling from any target
    pdf or a multiple of it. The implementation follows the description given
    in page 375 of David McKay's book "Information Theory, Inference, and
    Learning Algorithms" (2003).

    Parameters
    ----------
    ptarget : function with one input
        the target pdf to sample from or a multiple of it
    n_samples : int
        how many samples to get from the ptarget distribution
    x0 : array
        initial state for the MCMC procedure
    n_burnin : int
        how many samples to discard from the beginning of the chain generated
        by the slice sampling procedure. Usually the first samples are prone to
        non-stationary behavior and do not follow very well the target pdf
    thin : int
        thinning factor for the slice sampling procedure. MCMC samples are
        often correlated between them, so taking one sample every `thin`
        samples can help reducing this correlation. Note that this makes the
        algorithm actually sample `thin x n_samples` samples from the pdf, so
        expect the whole sampling procedure to take longer
    random_state : int, RandomState instance or None (default: 42)
        pass an int for reproducible output

    Returns
    -------
    samples : ndarray, shape (n_samples, n_dim)
        samples from the target pdf
    """

    if (n_samples <= 0) or (not isinstance(n_samples, int)):
        raise ValueError(
            f'n_samples must be a positive integer (Got {n_samples})')
    if (n_burnin <= 0) or (not isinstance(n_burnin, int)):
        raise ValueError(
            f'n_samples must be a positive integer (Got {n_burnin})')
    if (thin <= 0) or (not isinstance(thin, int)):
        raise ValueError(
            f'thin must be a positive integer (Got {thin})')

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
            xl_i = xt[i] - r*w
            xr_i = xt[i] + (1-r)*w
            while ptarget(xt + (xl_i - xt[i])*ei) > uprime_i:
                xl_i = xl_i - w
            while ptarget(xt + (xr_i - xt[i])*ei) > uprime_i:
                xr_i = xr_i + w

            # step 4 : loop
            while True:
                xprime_i = xl_i + (xr_i - xl_i) * rs.rand()
                Px = ptarget(xt + (xprime_i - xt[i])*ei)
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

    samples = np.array(samples)[(n_burnin*thin):][::thin]

    return samples


def _sample_parameter_r(n_samples, n_dim, sigma, random_state=42):
    """Sample the r parameters of a Riemannian Gaussian distribution

    Sample the logarithm of the eigenvalues of a SPD matrix following a
    Riemannian Gaussian distribution.

    See https://arxiv.org/pdf/1507.01760.pdf for the mathematical details

    Parameters
    ----------
    n_samples : int
        how many samples to generate
    n_dim : int
        dimensionality of the SPD matrices to be sampled
    sigma : float
        dispersion of the Riemannian Gaussian distribution
    random_state : int, RandomState instance or None (default: 42)
        pass an int for reproducible output

    Returns
    -------
    r_samples : ndarray (n_samples, n_dim)
        samples of the r parameters of the Riemannian Gaussian distribution
    """

    rs = check_random_state(random_state)
    x0 = rs.randn(n_dim)
    ptarget = partial(_pdf_r, sigma=sigma)
    r_samples = _slice_sampling(
        ptarget, n_samples=n_samples, x0=x0, random_state=random_state)

    return r_samples


def _sample_parameter_U(n_samples, n_dim, random_state=42):
    """Sample the U parameters of a Riemannian Gaussian distribution

    Sample the eigenvectors a SPD matrix following a Riemannian Gaussian
    distribution.

    See https://arxiv.org/pdf/1507.01760.pdf for the mathematical details

    Parameters
    ----------
    n_samples : int
        how many samples to generate
    n_dim : int
        dimensionality of the SPD matrices to be sampled
    random_state : int, RandomState instance or None (default: 42)
        pass an int for reproducible output

    Returns
    -------
    u_samples : ndarray (n_samples, n_dim)
        samples of the U parameters of the Riemannian Gaussian distribution
    """

    rs = check_random_state(random_state)
    u_samples = np.zeros((n_samples, n_dim, n_dim))
    for i in range(n_samples):
        A = rs.randn(n_dim, n_dim)
        Q, _ = np.linalg.qr(A)
        u_samples[i] = Q

    return u_samples


def sample_gaussian_spd_centered(n_samples, n_dim, sigma, random_state=42):
    """Sample a Riemannian Gaussian distribution centered at the Identity

    Sample SPD matrices from a Riemannian Gaussian distribution centered at the
    Identity, which has the role of the origin in the SPD manifold, and
    dispersion parametrized by sigma.

    See https://arxiv.org/pdf/1507.01760.pdf for the mathematical details

    Parameters
    ----------
    n_samples : int
        how many samples to generate
    n_dim : int
        dimensionality of the SPD matrices to be sampled
    sigma : float
        dispersion of the Riemannian Gaussian distribution
    random_state : int, RandomState instance or None (default: 42)
        pass an int for reproducible output

    Returns
    -------
    samples : ndarray (n_samples, n_dim, n_dim)
        samples of the Riemannian Gaussian distribution
    """

    samples_r = _sample_parameter_r(n_samples=n_samples,
                                    n_dim=n_dim,
                                    sigma=sigma,
                                    random_state=random_state)
    samples_U = _sample_parameter_U(n_samples=n_samples,
                                    n_dim=n_dim,
                                    random_state=random_state)

    samples = np.zeros((n_samples, n_dim, n_dim))
    for i in range(n_samples):
        Ui = samples_U[i]
        ri = samples_r[i]
        samples[i] = Ui.T @ np.diag(np.exp(ri)) @ Ui
        samples[i] = (samples[i] + samples[i].T) / 2.0  # ensure symmetry

    return samples


def sample_gaussian_spd(n_samples, mean, sigma, random_state=42):
    """Sample a Riemannian Gaussian distribution

    Sample SPD matrices from a Riemannian Gaussian distribution centered at
    mean and with dispersion parametrized by sigma. This distribution has been
    defined in Said et al. "Riemannian Gaussian Distributions on the space of
    symmetric positive definite matrices" (2016) and generalizes the notion of
    a Gaussian distribution to the space of SPD matrices. The sampling is based
    on a spectral factorization of SPD matrices in terms of their eigenvectors
    (U-parameters) and the log of the eigenvalues (r-parameters).

    See https://arxiv.org/pdf/1507.01760.pdf for more details

    Parameters
    ----------
    n_samples : int
        how many samples to generate
    mean : ndarray (n_dim, n_dim)
        center of the Riemannian Gaussian distribution
    sigma : float
        dispersion of the Riemannian Gaussian distribution
    random_state : int, RandomState instance or None (default: None)
        Pass an int for reproducible output across multiple function calls.

    Returns
    -------
    samples : ndarray, shape (n_samples, n_dim, n_dim)
        samples of the Riemannian Gaussian distribution
    """

    n_dim = mean.shape[0]
    samples_centered = sample_gaussian_spd_centered(
                        n_samples,
                        n_dim=n_dim,
                        sigma=sigma,
                        random_state=random_state)

    # apply the parallel transport to mean on each of the samples
    samples = np.zeros((n_samples, n_dim, n_dim))
    for i in range(n_samples):
        samples[i] = sqrtm(mean) @ samples_centered[i] @ sqrtm(mean)
        samples[i] = (samples[i] + samples[i].T) / 2.0  # ensure symmetry

    return samples


def generate_random_spd_matrix(n_dim, random_state=42):
    """Generate a random SPD matrix

    Parameters
    ----------
    n_dim : int
        dimensionality of the matrix to sample
    random_state : int, RandomState instance or None (default: 42)
        pass an int for reproducible output

    Returns
    -------
    C : ndarray, shape (n_dim, n_dim)
        random SPD matrix

    """
    rs = check_random_state(random_state)
    A = rs.randn(n_dim, n_dim)
    A = (A + A.T)/2
    _, Q = np.linalg.eig(A)
    w = rs.rand(n_dim)
    C = Q @ np.diag(w) @ Q.T

    return C
