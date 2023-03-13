from distutils.version import LooseVersion
import numpy as np


def check_version(library, min_version):
    """Check minimum library version required

    Parameters
    ----------
    library : str
        The library name to import. Must have a ``__version__`` property.
    min_version : str
        The minimum version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``

    Returns
    -------
    ok : bool
        True if the library exists with at least the specified version.

    Adapted from MNE-Python: http://github.com/mne-tools/mne-python
    """
    ok = True
    try:
        library = __import__(library)
    except ImportError:
        ok = False
    else:
        this_version = LooseVersion(library.__version__)
        if this_version < min_version:
            ok = False
    return ok


def check_weights(weights, n_weights, *, check_positivity=False):
    """Check weights.

    If input is None, output weights are equal.
    Strict positivity of weights can be checked.
    In any case, weights are normalized (sum equal to 1).

    Parameters
    ----------
    weights : None | ndarray, shape (n_weights,), default=None
        Input weights. If None, it provides equal weights.
    n_weights : int
        Number of weights to provide if None, or to check.
    check_positivity : bool, default=False
        Choose if strict positivity of weights is checked.

    Returns
    -------
    weights : ndarray, shape (n_weights,)
        Output checked weights.
    """
    if weights is None:
        weights = np.ones(n_weights)

    else:
        weights = np.asarray(weights)
        if weights.shape != (n_weights,):
            raise ValueError(
                "Weights do not have the good shape. Should be (%d,) but got "
                "%s." % (n_weights, weights.shape,)
            )
        if check_positivity and any(weights <= 0):
            raise ValueError("Weights must be strictly positive.")

    weights /= np.sum(weights)
    return weights
