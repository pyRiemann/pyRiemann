"""Backward-compat shim. Module moved to pyriemann.geometry.covariance."""
import sys
import warnings

from ..geometry import covariance as _moved

warnings.warn(
    "pyriemann.utils.covariance is deprecated and will be removed in 0.14.0; "
    "use pyriemann.geometry.covariance instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
