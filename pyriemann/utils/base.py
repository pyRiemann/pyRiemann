"""Backward-compatibility shim. Module moved to pyriemann.geometry.base."""
import sys

from pyriemann.geometry import base as _moved

sys.modules[__name__] = _moved
