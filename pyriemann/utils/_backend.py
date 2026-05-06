"""Backward-compat re-export of pyriemann.geometry._backend."""
import sys

from ..geometry import _backend as _moved

sys.modules[__name__] = _moved
