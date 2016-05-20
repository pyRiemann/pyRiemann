"""Mean covariance estimation."""
import numpy as np
from numpy import average, diag, eye, finfo, ones, zeros, log, log1p, exp, min, max
from numpy.linalg import inv, norm
from scipy.linalg import cholesky, schur
from .base import sqrtm, invsqrtm, logm, expm
from .ajd import ajd_pham
from .distance import distance_riemann


def _get_sample_weight(sample_weight, data):
    """Get the sample weights.

    If none provided, weights init to 1. otherwise, weights are normalized.
    """
    if sample_weight is None:
        sample_weight = ones(data.shape[0])
    if len(sample_weight) != data.shape[0]:
        raise ValueError("len of sample_weight must be equal to len of data.")
    sample_weight /= sample_weight.sum()
    return sample_weight


def mean_karcher(covmats, theta=None, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None):
    """Return Karcher mean using a Richardson-like iteration

    This iterative approach relies on Riemannian metric where the parameter
    theta may be chosen automatically, and the initial value is the
    arithmetic mean.    
    
    .. math::
            \mathbf{C} = \\arg\min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}

    This is the adaptation of the Matlab code proposed by Dario Bini and
    Bruno Iannazzo, http://bezout.dm.unipi.it/software/mmtoolbox/
    At least 3 times slower than mean_riemann, possible improvments.
            
    :param covmats: Covariance matrices set, (n_trials, n_channels, n_channels)
    :param theta: parameter of the iteration
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: maximum number of iteration, default 50
    :param init: covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample

    :returns: Karcher mean covariance matrix

    References
    ----------
    [1] D. A. Bini and B. Iannazzo, Computing the Karcher mean of symmetric
    positive definite matrices, to appear in Linear Algebra Appl., 2012.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    ni, ni_prev = 0., finfo(np.float64).max
    if init is None:
        C = covmats.mean(axis=0)
    else:
        C = init
    R = zeros(shape=(Nt, Ne, Ne))
    U = zeros(shape=(Nt, Ne, Ne))
    V = zeros(shape=(Nt, Ne))
    # X = C
    for h in range(Nt):
        R[h, :, :] = cholesky(covmats[h, :, :])

    for k in range(maxiter):
        try:
            Rc = cholesky(C)
        except ValueError:
            print ("[iteration %d] array must not contain infs or NaNs"%k)
            print (C)
            raise (ValueError, "Convergence error")
                
        iRc = inv(Rc)
        for h in range(Nt):
            Z = R[h, :, :].dot(iRc)
            Vz, U[h, :, :] = schur(Z.T.dot(Z))
            V[h, :] = diag(Vz)
        if theta == None:
            beta, gamma = 0., 0.
            for h in range(Nt):
                if np.isnan(beta):
                    print ('dh[%d]:'%h, dh)
                ch = max(V[h, :])/min(V[h, :])
                if ch == 1.:
                    dh = 0.
                elif abs(ch-1.) < 0.5:
                    dh = log1p(ch-1)/(ch-1)
                else:
                    dh = log(ch) / (ch-1)
                beta += dh
                gamma += ch*dh
                
            theta = 2/(gamma+beta)                

        S=zeros(shape=(Ne, Ne))
        for h in range(Nt):
            Sh = U[h, :, :].dot(diag(log(V[h, :]))).dot(U[h, :, :].T)
            S += (Sh + Sh.T)/2
        Vs, Us = schur(S)
        Z = diag(exp(diag(Vs*theta/2))).dot(Us.T).dot(Rc)
        C = Z.T.dot(Z)

        ni = max(abs(diag(Vs)))
        if (ni < norm(C)*tol) or ni > ni_prev:
            it = k
            break
        ni_prev = ni

        if k == maxiter:
            print ("Max iterations reached")
            it = k
    # return C, it, theta
    return C


def mean_alm():
    """Return Ando-Li-Mathias mean 

    Find the geometric mean recursively [1], generalizing from:
    
    .. math::
            \mathbf{C} = A^{\frac{1}{2}}(A^{-\frac{1}{2}}B^{\frac{1}{2}}A^{-\frac{1}{2}})^{\frac{1}{2}}A^{\frac{1}{2}}

    require a number of iterations.

    This is the adaptation of the Matlab code proposed by Dario Bini and
    Bruno Iannazzo, http://bezout.dm.unipi.it/software/mmtoolbox/
    At least 3 times slower than mean_riemann, possible improvments.
            
    :param covmats: Covariance matrices set, (n_trials, n_channels, n_channels)
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: maximum number of iteration, default 50
    :param init: covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample

    :returns: Karcher mean covariance matrix

    References
    ----------
    [1] T. Ando, C.-K. Li and R. Mathias, "Geometric Means", Linear Algebra
    Appl. 385 (2004), 305-334.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    
  

def mean_riemann(covmats, tol=10e-9, maxiter=50, init=None,
                 sample_weight=None):
    """Return the mean covariance matrix according to the Riemannian metric.

    The procedure is similar to a gradient descent minimizing the sum of
    riemannian distance to the mean.

    .. math::
            \mathbf{C} = \\arg\min{(\sum_i \delta_R ( \mathbf{C} , \mathbf{C}_i)^2)}

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the gradient descent. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample
    :returns: the mean covariance matrix

    """
    # init
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = covmats.mean(axis=0)
    else:
        C = init
    k, nu = 0, 1.0
    tau = finfo(np.float64).max
    crit = finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter) and (nu > tol):
        k = k + 1
        C12 = sqrtm(C)
        Cm12 = invsqrtm(C)
        J = zeros(shape=(Ne, Ne))

        for index in range(Nt):
            tmp = Cm12.dot(covmats[index, :, :]).dot(Cm12)
            J += sample_weight[index] * logm(tmp)

        crit = norm(J, ord='fro')
        h = nu * crit
        C = C12.dot(expm(nu * J)).dot(C12)
        if h < tau:
            nu = 0.95 * nu
            tau = h
        else:
            nu = 0.5 * nu

    return C


def mean_logeuclid(covmats, sample_weight=None):
    """Return the mean covariance matrix according to the log-euclidean metric.

    .. math::
            \mathbf{C} = \exp{(\\frac{1}{N} \sum_i \log{\mathbf{C}_i})}

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    T = zeros(shape=(Ne, Ne))
    for index in range(Nt):
        T += sample_weight[index] * logm(covmats[index, :, :])
    C = expm(T)

    return C


def mean_logdet(covmats, tol=10e-5, maxiter=50, init=None, sample_weight=None):
    """Return the mean covariance matrix according to the logdet metric.

    This is an iterative procedure where the update is:

    .. math::
            \mathbf{C} = \left(\sum_i \left( 0.5 \mathbf{C} + 0.5 \mathbf{C}_i \\right)^{-1} \\right)^{-1}

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = covmats.mean(axis=0)
    else:
        C = init
    k = 0
    crit = finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1

        J = zeros(shape=(Ne, Ne))

        for index, Ci in enumerate(covmats):
            J += sample_weight[index] * inv(0.5 * Ci + 0.5 * C)

        Cnew = inv(J)
        crit = norm(Cnew - C, ord='fro')

        C = Cnew
    if k == maxiter:
        print('Max iter reach')
    return C


def mean_wasserstein(covmats, tol=10e-4, maxiter=50, init=None,
                     sample_weight=None):
    """Return the mean covariance matrix according to the wasserstein metric.

    This is an iterative procedure where the update is [1]:

    .. math::
            \mathbf{K} = \left(\sum_i \left( \mathbf{K} \mathbf{C}_i \mathbf{K} \\right)^{1/2} \\right)^{1/2}

    with :math:`\mathbf{K} = \mathbf{C}^{1/2}`.

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param init: A covariance matrix used to initialize the iterative procedure. If None the Arithmetic mean is used
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    References
    ----------
    [1] Barbaresco, F. "Geometric Radar Processing based on Frechet distance:
    Information geometry versus Optimal Transport Theory", Radar Symposium
    (IRS), 2011 Proceedings International.
    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    if init is None:
        C = covmats.mean(axis=0)
    else:
        C = init
    k = 0
    K = sqrtm(C)
    crit = finfo(np.float64).max
    # stop when J<10^-9 or max iteration = 50
    while (crit > tol) and (k < maxiter):
        k = k + 1

        J = zeros(shape=(Ne, Ne))

        for index, Ci in enumerate(covmats):
            tmp = K.dot(Ci).dot(K)
            J += sample_weight[index] * sqrtm(tmp)

        Knew = sqrtm(J)
        crit = norm(Knew - K, ord='fro')
        K = Knew
    if k == maxiter:
        print('Max iter reach')
    C = K.dot(K)
    return C


def mean_euclid(covmats, sample_weight=None):
    """Return the mean covariance matrix according to the euclidean metric :

    .. math::
            \mathbf{C} = \\frac{1}{N} \sum_i \mathbf{C}_i

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    """
    return average(covmats, axis=0, weights=sample_weight)


def mean_ale(covmats, tol=10e-7, maxiter=50, sample_weight=None):
    """Return the mean covariance matrix according using the AJD-based
    log-Euclidean Mean (ALE). See [1].

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param tol: the tolerance to stop the gradient descent
    :param maxiter: The maximum number of iteration, default 50
    :param sample_weight: the weight of each sample

    :returns: the mean covariance matrix

    Notes
    -----
    .. versionadded:: 0.2.4

    References
    ----------
    [1] M. Congedo, B. Afsari, A. Barachant, M. Moakher, 'Approximate Joint
    Diagonalization and Geometric Mean of Symmetric Positive Definite
    Matrices', PLoS ONE, 2015

    """
    sample_weight = _get_sample_weight(sample_weight, covmats)
    Nt, Ne, Ne = covmats.shape
    crit = np.inf
    k = 0

    # init with AJD
    B, _ = ajd_pham(covmats)
    while (crit > tol) and (k < maxiter):
        k += 1
        J = zeros(shape=(Ne, Ne))

        for index, Ci in enumerate(covmats):
            tmp = logm(B.T.dot(Ci).dot(B))
            J += sample_weight[index] * tmp

        update = diag(diag(expm(J)))
        B = B.dot(invsqrtm(update))

        crit = distance_riemann(eye(Ne), update)

    A = inv(B)

    J = zeros(shape=(Ne, Ne))
    for index, Ci in enumerate(covmats):
        tmp = logm(B.T.dot(Ci).dot(B))
        J += sample_weight[index] * tmp

    C = A.T.dot(expm(J)).dot(A)
    return C


def mean_identity(covmats, sample_weight=None):
    """Return the identity matrix corresponding to the covmats sit size

    .. math::
            \mathbf{C} = \mathbf{I}_d

    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :returns: the identity matrix of size Nchannels

    """
    C = eye(covmats.shape[1])
    return C


def mean_covariance(covmats, metric='riemann', sample_weight=None, *args):
    """Return the mean covariance matrix according to the metric


    :param covmats: Covariance matrices set, Ntrials X Nchannels X Nchannels
    :param metric: the metric (Default value 'riemann'), can be : 'riemann' , 'logeuclid' , 'euclid' , 'logdet', 'indentity', 'wasserstein'
    :param sample_weight: the weight of each sample
    :param args: the argument passed to the sub function
    :returns: the mean covariance matrix

    """
    options = {'riemann': mean_riemann,
               'logeuclid': mean_logeuclid,
               'euclid': mean_euclid,
               'identity': mean_identity,
               'logdet': mean_logdet,
               'wasserstein': mean_wasserstein,
               'ale': mean_ale,
               'karcher': mean_karcher}
    C = options[metric](covmats, sample_weight=sample_weight, *args)
    return C
