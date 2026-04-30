"""Backward-compatibility shim. Module moved to pyriemann.geometry.median."""
import sys

from pyriemann.geometry import median as _moved

sys.modules[__name__] = _moved
