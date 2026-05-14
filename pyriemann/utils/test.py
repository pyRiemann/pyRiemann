"""Backward-compatibility shim. Module moved to pyriemann.geometry.test."""
import sys
import warnings

from ..geometry import test as _moved

warnings.warn(
    "pyriemann.utils.test is deprecated and will be removed in 0.14.0; "
    "use pyriemann.geometry.test instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
