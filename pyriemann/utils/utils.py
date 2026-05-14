"""Backward-compatibility shim. Module renamed to pyriemann.utils._check."""
import sys
import warnings

from . import _check as _moved

warnings.warn(
    "pyriemann.utils.utils is deprecated and will be removed in 0.14.0; "
    "use pyriemann.utils._check instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
