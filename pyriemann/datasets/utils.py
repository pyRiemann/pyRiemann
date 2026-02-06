"""Utility functions for dataset management."""

import os


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
        base = os.path.join(os.path.expanduser("~"), "pyriemann_data")
    base = os.path.abspath(base)

    if dataset_name is not None:
        return os.path.join(base, dataset_name)
    return base
