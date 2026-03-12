"""Inner product for SPD/HPD matrices."""

import numpy as np

from .base import ctranspose, ddlogm, invsqrtm
from .utils import check_function


def innerproduct_euclid(X, Y, *args):
    r"""Euclidean inner product.

    Euclidean inner product :math:`\mathbf{g}`
    between matrices :math:`\mathbf{X}` and :math:`\mathbf{Y}` is:

    .. math::
        \mathbf{g}(\mathbf{X}, \mathbf{Y}) = \text{tr}(\mathbf{X}^H \mathbf{Y})

    Parameters
    ----------
    X : ndarray, shape (..., n, m)
        First  matrices.
    Y : ndarray, shape (..., n, m) | None
        Second matrices.
        If None, Y is set to X, giving the squared norm of X.

    Returns
    -------
    G : float or ndarray, shape (...,)
        Euclidean inner product between X and Y.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct
    """
    if Y is None:
        Y = X
    G = _apply_inner_product(ctranspose(X), Y)
    return G


def innerproduct_logeuclid(X, Y, Cref):
    r"""Log-Euclidean inner product.

    Log-Euclidean inner product :math:`\mathbf{g}`
    between SPD/HPD matrices :math:`\mathbf{X}` and :math:`\mathbf{Y}`
    on tangent space at :math:`\mathbf{C}_\text{ref}` is [1]_:

    .. math::
        \mathbf{g}_{\mathbf{C}_\text{ref}}(\mathbf{X}, \mathbf{Y}) =
            \text{tr} \left(
            [D_{\mathbf{C}_\text{ref}} \log](X)^*
            [D_{\mathbf{C}_\text{ref}} \log](Y)
            \right)

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        First SPD/HPD matrices.
    Y : ndarray, shape (..., n, n) | None
        Second SPD/HPD matrices.
        If None, Y is set to X, giving the squared norm of X.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.

    Returns
    -------
    G : float or ndarray, shape (...,)
        Log-Euclidean inner product between X and Y.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct

    References
    ----------
    .. [1] `Geometric means in a novel vector space structure on symmetric
        positive-definite matrices
        <https://epubs.siam.org/doi/abs/10.1137/050637996>`_
        V. Arsigny, P. Fillard, X. Pennec, N. Ayache.
        SIAM J Matrix Anal Appl, 2007, 29 (1), pp. 328-347
    """
    if Y is None:
        Y = X
    G = _apply_inner_product(ddlogm(X, Cref).conj(), ddlogm(Y, Cref))
    return G


def innerproduct_riemann(X, Y, Cref):
    r"""Affine-invariant Riemannian inner product.

    Affine-invariant Riemannian inner product :math:`\mathbf{g}`
    between SPD/HPD matrices :math:`\mathbf{X}` and :math:`\mathbf{Y}`
    on tangent space at :math:`\mathbf{C}_\text{ref}` is [1]_:

    .. math::
        \mathbf{g}_{\mathbf{C}_\text{ref}}(\mathbf{X}, \mathbf{Y}) =
            \text{tr} \left(
            (\mathbf{C}_\text{ref}^{-1/2} \mathbf{X}
            \mathbf{C}_\text{ref}^{-1/2})^*
            \mathbf{C}_\text{ref}^{-1/2} \mathbf{Y}
            \mathbf{C}_\text{ref}^{-1/2}
            \right)

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        First SPD/HPD matrices.
    Y : ndarray, shape (..., n, n) | None
        Second SPD/HPD matrices.
        If None, Y is set to X, giving the squared norm of X.
    Cref : ndarray, shape (n, n)
        Reference SPD/HPD matrix.

    Returns
    -------
    G : float or ndarray, shape (...,)
        Affine-invariant Riemannian inner product between X and Y.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct

    References
    ----------
    .. [1] `A metric for covariance matrices
        <https://www.ipb.uni-bonn.de/pdfs/Forstner1999Metric.pdf>`_
        W. Förstner & B. Moonen.
        Geodesy-the Challenge of the 3rd Millennium, 2003
    """
    if Y is None:
        Y = X
    Cm12 = invsqrtm(Cref)
    G = _apply_inner_product((Cm12 @ X @ Cm12).conj(), Cm12 @ Y @ Cm12)
    return G


###############################################################################


def _apply_inner_product(Xt, Y):
    # product G = trace(Xt @ Y)
    G = np.einsum("...mn,...nm->...", Xt, Y, optimize=True)

    if G.size == 1:
        return G.item()
    else:
        return G


innerproduct_functions = {
    "euclid": innerproduct_euclid,
    "logeuclid": innerproduct_logeuclid,
    "riemann": innerproduct_riemann,
}


def innerproduct(X, Y, Cref, metric="riemann"):
    r"""Inner product according to a specified metric.

    It calculates the inner product between matrices :math:`\mathbf{X}` and
    :math:`\mathbf{Y}` on the tangent space at :math:`\mathbf{C}_\text{ref}`,
    according to a specified metric.

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        First matrices.
    Y : ndarray, shape (..., n, n) | None
        Second matrices.
        If None, Y is set to X, giving the squared norm of X.
    Cref : ndarray, shape (n, n) | None
        Reference matrix.
    metric : string | callable, default="riemann"
        Metric used for inner product, can be:
        "euclid", "logeuclid", "riemann", or a callable function.

    Returns
    -------
    K : float or ndarray, shape (...,)
        Inner product between X and Y.

    Notes
    -----
    .. versionadded:: 0.11

    See Also
    --------
    innerproduct_euclid
    innerproduct_logeuclid
    innerproduct_riemann
    """
    innerproduct_function = check_function(metric, innerproduct_functions)
    K = innerproduct_function(X, Y, Cref)
    return K
