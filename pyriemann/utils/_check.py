import inspect

from ..geometry._helpers import (  # noqa: F401
    check_function,
    check_init,
    check_like,
    check_matrix_pair,
    check_weights,
)


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
        raise TypeError(f"metric must be str or dict, but got {type(metric)}")


def check_param_in_func(param, func):
    """Check if a parameter is an argument of a function.

    Parameters
    ----------
    param : str
        Name of the parameter to check.
    func : callable
        Function to check.

    Returns
    -------
    ret : bool
        True if param is an argument of a function, else False.

    Notes
    -----
    .. versionadded:: 0.11
    """
    sig = inspect.signature(func).parameters
    return (param in sig)
