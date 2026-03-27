"""Riemannian geometry for SPD/HPD matrices."""

from .base import (
    ctranspose, expm, invsqrtm, logm, powm, sqrtm,
    nearest_sym_pos_def, ddexpm, ddlogm,
)
from .covariance import (
    covariances, covariance_mest, covariance_sch, covariance_scm,
    covariances_EP, covariances_X, block_covariances,
    cross_spectrum, cospectrum, coherence, normalize, get_nondiag_weight,
)
from .distance import (
    distance, distance_chol, distance_euclid, distance_harmonic,
    distance_kullback, distance_kullback_right, distance_kullback_sym,
    distance_logchol, distance_logdet, distance_logeuclid,
    distance_poweuclid, distance_riemann, distance_thompson,
    distance_wasserstein, pairwise_distance, distance_mahalanobis,
)
from .geodesic import (
    geodesic, geodesic_chol, geodesic_euclid, geodesic_logchol,
    geodesic_logeuclid, geodesic_riemann, geodesic_thompson,
    geodesic_wasserstein,
)
from .mean import (
    gmean, mean_ale, mean_alm, mean_chol, mean_euclid, mean_harmonic,
    mean_kullback_sym, mean_logchol, mean_logdet, mean_logeuclid,
    mean_power, mean_poweuclid, mean_riemann, mean_thompson,
    mean_wasserstein, maskedmean_riemann, nanmean_riemann,
)
from .median import median_euclid, median_riemann
from .tangentspace import (
    exp_map, exp_map_euclid, exp_map_logchol, exp_map_logeuclid,
    exp_map_riemann, exp_map_wasserstein,
    log_map, log_map_euclid, log_map_logchol, log_map_logeuclid,
    log_map_riemann, log_map_wasserstein,
    upper, unupper, tangent_space, untangent_space,
    innerproduct, innerproduct_euclid, innerproduct_logeuclid,
    innerproduct_riemann, norm,
    transport, transport_euclid, transport_logchol,
    transport_logeuclid, transport_riemann,
)
from .ajd import ajd, ajd_pham, rjd, uwedge


__all__ = [
    # base
    "ctranspose", "expm", "invsqrtm", "logm", "powm", "sqrtm",
    "nearest_sym_pos_def", "ddexpm", "ddlogm",
    # covariance
    "covariances", "covariance_mest", "covariance_sch", "covariance_scm",
    "covariances_EP", "covariances_X", "block_covariances",
    "cross_spectrum", "cospectrum", "coherence", "normalize",
    "get_nondiag_weight",
    # distance
    "distance", "distance_chol", "distance_euclid", "distance_harmonic",
    "distance_kullback", "distance_kullback_right", "distance_kullback_sym",
    "distance_logchol", "distance_logdet", "distance_logeuclid",
    "distance_poweuclid", "distance_riemann", "distance_thompson",
    "distance_wasserstein", "pairwise_distance", "distance_mahalanobis",
    # geodesic
    "geodesic", "geodesic_chol", "geodesic_euclid", "geodesic_logchol",
    "geodesic_logeuclid", "geodesic_riemann", "geodesic_thompson",
    "geodesic_wasserstein",
    # mean
    "gmean", "mean_ale", "mean_alm", "mean_chol", "mean_euclid",
    "mean_harmonic", "mean_kullback_sym", "mean_logchol", "mean_logdet",
    "mean_logeuclid", "mean_power", "mean_poweuclid", "mean_riemann",
    "mean_thompson", "mean_wasserstein", "maskedmean_riemann",
    "nanmean_riemann",
    # median
    "median_euclid", "median_riemann",
    # tangentspace
    "exp_map", "exp_map_euclid", "exp_map_logchol", "exp_map_logeuclid",
    "exp_map_riemann", "exp_map_wasserstein",
    "log_map", "log_map_euclid", "log_map_logchol", "log_map_logeuclid",
    "log_map_riemann", "log_map_wasserstein",
    "upper", "unupper", "tangent_space", "untangent_space",
    "innerproduct", "innerproduct_euclid", "innerproduct_logeuclid",
    "innerproduct_riemann", "norm",
    "transport", "transport_euclid", "transport_logchol",
    "transport_logeuclid", "transport_riemann",
    # ajd
    "ajd", "ajd_pham", "rjd", "uwedge",
]
