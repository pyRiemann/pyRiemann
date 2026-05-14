"""Backward-compatibility shim. Module moved to pyriemann.geometry.base."""
import sys
import warnings

from ..geometry import base as _moved

warnings.warn(
    "pyriemann.utils.base is deprecated and will be removed in 0.14.0; "
    "use pyriemann.geometry.base instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
