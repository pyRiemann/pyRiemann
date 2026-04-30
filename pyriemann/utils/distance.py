"""Backward-compatibility shim. Module moved to pyriemann.geometry.distance."""
import sys

from pyriemann.geometry import distance as _moved

sys.modules[__name__] = _moved
