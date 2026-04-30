"""Backward-compatibility shim. Module moved to pyriemann.geometry.tangentspace."""
import sys

from pyriemann.geometry import tangentspace as _moved

sys.modules[__name__] = _moved
