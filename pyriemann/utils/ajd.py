"""Backward-compatibility shim. Module moved to pyriemann.geometry.ajd."""
import sys

from pyriemann.geometry import ajd as _moved

sys.modules[__name__] = _moved
