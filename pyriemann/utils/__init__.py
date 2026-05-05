from .docs import deprecated  # noqa: F401


__all__ = [
    "gmean",
    "mean_ale",
    "mean_alm",
    "mean_euclid",
    "mean_harmonic",
    "mean_kullback_sym",
    "mean_logdet",
    "mean_logeuclid",
    "mean_power",
    "mean_riemann",
    "mean_wasserstein",
    "maskedmean_riemann",
    "nanmean_riemann",
    "median_euclid",
    "median_riemann",
]

_LAZY_MAP = {
    "gmean": "pyriemann.geometry.mean",
    "mean_ale": "pyriemann.geometry.mean",
    "mean_alm": "pyriemann.geometry.mean",
    "mean_euclid": "pyriemann.geometry.mean",
    "mean_harmonic": "pyriemann.geometry.mean",
    "mean_kullback_sym": "pyriemann.geometry.mean",
    "mean_logdet": "pyriemann.geometry.mean",
    "mean_logeuclid": "pyriemann.geometry.mean",
    "mean_power": "pyriemann.geometry.mean",
    "mean_riemann": "pyriemann.geometry.mean",
    "mean_wasserstein": "pyriemann.geometry.mean",
    "maskedmean_riemann": "pyriemann.geometry.mean",
    "nanmean_riemann": "pyriemann.geometry.mean",
    "median_euclid": "pyriemann.geometry.median",
    "median_riemann": "pyriemann.geometry.median",
}


def __getattr__(name):
    # Lazy re-exports from pyriemann.geometry to avoid a circular import:
    # geometry.mean depends on pyriemann.utils._backend, which would otherwise
    # require this __init__ to finish loading before geometry.mean does.
    target = _LAZY_MAP.get(name)
    if target is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    import importlib
    return getattr(importlib.import_module(target), name)
