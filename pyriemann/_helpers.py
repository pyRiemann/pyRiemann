"""Backward-compat re-export of pyriemann.geometry._helpers."""
import sys

from .geometry import _helpers as _moved

sys.modules[__name__] = _moved
