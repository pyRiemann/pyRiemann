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

_MEAN_NAMES = {
    "gmean", "mean_ale", "mean_alm", "mean_euclid", "mean_harmonic",
    "mean_kullback_sym", "mean_logdet", "mean_logeuclid", "mean_power",
    "mean_riemann", "mean_wasserstein", "maskedmean_riemann",
    "nanmean_riemann",
}

_MEDIAN_NAMES = {"median_euclid", "median_riemann"}


def __getattr__(name):
    if name in _MEAN_NAMES:
        from pyriemann.geometry import mean as _mean
        return getattr(_mean, name)
    if name in _MEDIAN_NAMES:
        from pyriemann.geometry import median as _median
        return getattr(_median, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
