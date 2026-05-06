"""Backward-compatibility shim. Module moved to pyriemann.geometry.geodesic."""
import sys
import warnings

from ..geometry import geodesic as _moved

warnings.warn(
    "pyriemann.utils.geodesic is deprecated and will be removed in "
    "0.14.0; use pyriemann.geometry.geodesic instead.",
    DeprecationWarning,
    stacklevel=2,
)

sys.modules[__name__] = _moved
