"""Utility functions for dataset management."""

import os
from pathlib import Path


def get_data_path(dataset_name=None):
    """Get the base path for pyRiemann datasets.

    Resolves the root directory for dataset storage. Checks the
    ``PYRIEMANN_DATA_PATH`` environment variable first, then falls back
    to ``~/pyriemann_data``.

    Parameters
    ----------
    dataset_name : str | None, default=None
        Optional dataset subdirectory name. When provided, it is appended
        to the base path.

    Returns
    -------
    path : str
        Absolute path to the dataset directory.
    """
    base = os.environ.get("PYRIEMANN_DATA_PATH")
    if base is None:
        base = Path.home() / "pyriemann_data"
    else:
        base = Path(base)
    base = base.resolve()

    if dataset_name is not None:
        return str(base / dataset_name)
    return str(base)
