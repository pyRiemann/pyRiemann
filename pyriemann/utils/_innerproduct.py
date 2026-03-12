"""Inner product for SPD matrices."""

import numpy as np

from .base import ctranspose, ddlogm, invsqrtm
from .utils import check_function


def innerproduct_euclid(X, Y, *args):
    r"""Euclidean inner product.

    Euclidean inner product :math:`\mathbf{g}` between two matrices
    :math:`\mathbf{X}` and :math:`\mathbf{Y}` in
    :math:`\mathbb{R}^{n \times m}` is:

    .. math::
        \mathbf{g}(\mathbf{X}, \mathbf{Y}) = \text{tr}(\mathbf{X}^T \mathbf{Y})

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

    Log-Euclidean inner product :math:`\mathbf{g}` between two SPD matrices
    :math:`\mathbf{X}` and :math:`\mathbf{Y}` in
    :math:`\mathbb{R}^{n \times n}` on tangent space at
    :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{g}_{\mathbf{C}_\text{ref}}(\mathbf{X}, \mathbf{Y}) =
            \text{tr} \left(
            [D_{\mathbf{C}_\text{ref}} \log](X)
            [D_{\mathbf{C}_\text{ref}} \log](Y)
            \right)

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        First set of SPD matrices.
    Y : ndarray, shape (..., n, n) | None
        Second set of SPD matrices.
        If None, Y is set to X, giving the squared norm of X.
    Cref : ndarray, shape (n, n)
        Reference SPD matrix.

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
    """
    if Y is None:
        Y = X
    G = _apply_inner_product(ddlogm(X, Cref), ddlogm(Y, Cref))
    return G


def innerproduct_riemann(X, Y, Cref):
    r"""Affine-invariant Riemannian inner product.

    Affine-invariant Riemannian inner product :math:`\mathbf{g}` between two
    SPD matrices :math:`\mathbf{X}` and :math:`\mathbf{Y}` in
    :math:`\mathbb{R}^{n \times n}` on tangent space at
    :math:`\mathbf{C}_\text{ref}` is:

    .. math::
        \mathbf{g}_{\mathbf{C}_\text{ref}}(\mathbf{X}, \mathbf{Y}) =
            \text{tr} \left(
            \mathbf{C}_\text{ref}^{-1/2} \mathbf{X} \mathbf{C}_\text{ref}^{-1}
            \mathbf{Y} \mathbf{C}_\text{ref}^{-1/2}
            \right)

    Parameters
    ----------
    X : ndarray, shape (..., n, n)
        First SPD matrices.
    Y : ndarray, shape (..., n, n) | None
        Second SPD matrices.
        If None, Y is set to X, giving the squared norm of X.
    Cref : ndarray, shape (n, n)
        Reference SPD matrix.

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
    """
    if Y is None:
        Y = X
    Cm12 = invsqrtm(Cref)
    G = _apply_inner_product(Cm12 @ X @ Cm12, Cm12 @ Y @ Cm12)
    return G


###############################################################################


def _apply_inner_product(Xt, Y):
    # product G = trace(Xt @ Y)
    G = np.einsum("...mn,...nm->...", Xt, Y, optimize=True)
    return G


innerproduct_functions = {
    "euclid": innerproduct_euclid,
    "logeuclid": innerproduct_logeuclid,
    "riemann": innerproduct_riemann,
}


def innerproduct(X, Y, Cref, metric="riemann"):
    r"""Inner product according to a specified metric.

    It calculates the kernel matrix :math:`\mathbf{K}` of pairwise inner
    products of two sets :math:`\mathbf{X}` and :math:`\mathbf{Y}`
    of matrices on the tangent space at :math:`\mathbf{C}_\text{ref}`,
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
