"""Backward-compatible re-exports from pyriemann.geometry.tangentspace."""
from pyriemann.geometry.tangentspace import *  # noqa: F401,F403
from pyriemann.geometry.tangentspace import (  # noqa: F401
    _check_dimensions, _apply_inner_product,
    exp_map_functions, log_map_functions,
    innerproduct_functions, transport_functions,
)
