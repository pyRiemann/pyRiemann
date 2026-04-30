"""Backward-compatibility shim. Module moved to pyriemann.geometry.geodesic."""
import sys

from pyriemann.geometry import geodesic as _moved

sys.modules[__name__] = _moved
