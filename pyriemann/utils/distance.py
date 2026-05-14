"""Backward-compatibility shim. Module moved to pyriemann.geometry.distance."""
import sys
import warnings

from ..geometry import distance as _moved

warnings.warn(
    "pyriemann.utils.distance is deprecated and will be removed in 0.14.0; "
    "use pyriemann.geometry.distance instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
