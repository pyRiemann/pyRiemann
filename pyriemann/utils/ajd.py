"""Backward-compatibility shim. Module moved to pyriemann.geometry.ajd."""
import sys
import warnings

from ..geometry import ajd as _moved

warnings.warn(
    "pyriemann.utils.ajd is deprecated and will be removed in a future "
    "release; use pyriemann.geometry.ajd instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
