"""Backward-compatibility shim. Module moved to pyriemann.geometry.mean."""
import sys
import warnings

from ..geometry import mean as _moved

warnings.warn(
    "pyriemann.utils.mean is deprecated and will be removed in 0.14.0; "
    "use pyriemann.geometry.mean instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
