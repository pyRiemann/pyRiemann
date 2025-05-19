import operator as operator_module
import re
import warnings

import numpy as np


def _strip_dev(version):
    exp = r"^([0-9]+(?:\.[0-9]+)*)\.?(?:dev|rc|\+)[0-9+a-g\.\-]+$"
    match = re.match(exp, version)
    return match.groups()[0] if match is not None else version


def _compare_version(version_a, operator, version_b):
    from packaging.version import parse

    mapping = {
        "<": "lt", "<=": "le", "==": "eq", "!=": "ne", ">=": "ge", ">": "gt"
    }
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("ignore")
        ver_a = parse(version_a)
        ver_b = parse(version_b)
        return getattr(operator_module, mapping[operator])(ver_a, ver_b)


def check_version(library, min_version, strip=True):
    """Check minimum library version required

    Parameters
    ----------
    library : str
        The library name to import. Must have a ``__version__`` property.
    min_version : str
        The minimum version string. Anything that matches
        ``'(\\d+ | [a-z]+ | \\.)'``
    strip : bool
        If True (default), then PEP440 development markers like ``.devN``
        will be stripped from the version. This makes it so that
        ``check_version('mne', '1.1')`` will be ``True`` even when on version
        ``'1.1.dev0'`` (prerelease/dev version). This option is provided for
        backward compatibility with the behavior of ``LooseVersion``, and
        diverges from how modern parsing in ``packaging.version.parse`` works.

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
        check_version = min_version and min_version != "0.0"
        version = library.__version__
        if strip:
            version = _strip_dev(version)
        if check_version and _compare_version(version, "<", min_version):
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

    Notes
    -----
    .. versionadded:: 0.4
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


def check_metric(metric, expected_keys=["mean", "distance"]):
    """Check metric argument.

    Parameters
    ----------
     metric : string | dict
        Metric to check in the algorithm: it can be a string, or a dictionary
        defining different metrics for the different steps of the algorithm.
        Typical usecase is to pass "logeuclid" metric for the "mean" in order
        to boost the computional speed, and "riemann" for the "distance" in
        order to keep the good sensitivity for the classification.
     expected_keys : list of str, default=["mean", "distance"]
        Names of the steps of the algorithm requiring a metric argument.

    Returns
    -------
     metric : list of str
        Metrics for each expected key.

    Notes
    -----
    .. versionadded:: 0.6
    """
    if isinstance(metric, str):
        return [metric] * len(expected_keys)

    elif isinstance(metric, dict):
        if not all(k in metric.keys() for k in expected_keys):
            raise KeyError(
                f"metric must contain {expected_keys}, but got {metric.keys()}"
            )

        return [metric[k] for k in expected_keys]

    else:
        raise TypeError("metric must be str or dict, but got {type(metric)}")


def check_function(fun, functions):
    """Check the function to use.

    Parameters
    ----------
    fun : string | callable
        Function to check.
        If string, it must be one of the keys of ``functions``.
        If callable, it can be a function defined in API or by the user.
        In the latter case, the signature of the function as to match the ones
        defined in ``functions``. This is the user responsibility to ensure
        this, and will not be checked.
    functions : dict
        Functions available in API, used only when ``fun`` is a string.

    Returns
    -------
    fun : callable
        Function to use.

    Notes
    -----
    .. versionadded:: 0.6
    """
    if isinstance(fun, str):
        if fun not in functions.keys():
            raise ValueError(f"Unknown function name '{fun}'. Must be one of "
                             f"{' '.join(functions.keys())}")
        else:
            fun = functions[fun]
    elif not hasattr(fun, '__call__'):
        raise ValueError("Argument must be a string or a callable "
                         f"(Got {type(fun)}).")
    return fun


def check_init(init, n):
    """Check the initial matrix.

    Parameters
    ----------
    init : ndarray, shape (n, n)
        A square matrix used to initialize the algorithm.
    n : int
        Expected dimension of the matrix.

    Returns
    -------
    init : ndarray, shape (n, n)
        The checked square matrix used to initialize the algorithm.

    Notes
    -----
    .. versionadded:: 0.8
    """
    init = np.asarray(init)
    if init.shape != (n, n):
        raise ValueError(
            "Init matrix does not have the good shape. "
            f"Should be ({n},{n}) but got {init.shape}."
        )
    return init
