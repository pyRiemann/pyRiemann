"""Backward-compatibility shim. Module moved to pyriemann.geometry.covariance."""
import sys

from pyriemann.geometry import covariance as _moved

sys.modules[__name__] = _moved
