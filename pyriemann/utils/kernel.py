"""Backward-compatibility shim. Module moved to pyriemann.geometry.kernel."""
import sys
import warnings

from ..geometry import kernel as _moved

warnings.warn(
    "pyriemann.utils.kernel is deprecated and will be removed in a future "
    "release; use pyriemann.geometry.kernel instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
