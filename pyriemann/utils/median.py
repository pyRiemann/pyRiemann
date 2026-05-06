"""Backward-compatibility shim. Module moved to pyriemann.geometry.median."""
import sys
import warnings

from ..geometry import median as _moved

warnings.warn(
    "pyriemann.utils.median is deprecated and will be removed in 0.14.0; "
    "use pyriemann.geometry.median instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
