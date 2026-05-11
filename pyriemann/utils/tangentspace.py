"""Backward-compat shim. Moved to pyriemann.geometry.tangentspace."""
import sys
import warnings

from ..geometry import tangentspace as _moved

warnings.warn(
    "pyriemann.utils.tangentspace is deprecated and will be removed in 0.14.0;"
    " use pyriemann.geometry.tangentspace instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
