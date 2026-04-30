"""Backward-compatibility shim. Module moved to pyriemann.geometry.mean."""
import sys

from pyriemann.geometry import mean as _moved

sys.modules[__name__] = _moved
