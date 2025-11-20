.. _api_ref:

=============
API reference
=============

SPD Matrices Estimation
-----------------------
.. _estimation_api:
.. currentmodule:: pyriemann.estimation

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Covariances
    ERPCovariances
    XdawnCovariances
    BlockCovariances
    CrossSpectra
    CoSpectra
    Coherences
    TimeDelayCovariances
    Kernels
    Shrinkage

Embedding
---------
.. _embedding_api:
.. currentmodule:: pyriemann.embedding

.. autosummary::
    :toctree: generated/

    locally_linear_embedding
    barycenter_weights

    :template: class.rst

    SpectralEmbedding
    LocallyLinearEmbedding
    TSNE

Classification
--------------
.. _classification_api:
.. currentmodule:: pyriemann.classification

.. autosummary::
    :toctree: generated/
    :template: class.rst

    MDM
    FgMDM
    TSClassifier
    KNearestNeighbor
    SVC
    MeanField

.. autosummary::
    :toctree: generated/
    :template: function.rst

    class_distinctiveness

Regression
--------------
.. _regression_api:
.. currentmodule:: pyriemann.regression

.. autosummary::
    :toctree: generated/
    :template: class.rst

    KNearestNeighborRegressor
    SVR

Clustering
------------------
.. _clustering_api:
.. currentmodule:: pyriemann.clustering

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Kmeans
    KmeansPerClassTransform
    MeanShift
    Potato
    PotatoField

Tangent Space
------------------
.. _tangentspace_api:
.. currentmodule:: pyriemann.tangentspace

.. autosummary::
    :toctree: generated/
    :template: class.rst

    TangentSpace
    FGDA

Spatial Filtering
------------------
.. _spatialfilter_api:
.. currentmodule:: pyriemann.spatialfilters

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Xdawn
    CSP
    SPoC
    BilinearFilter
    AJDC

Preprocessing
-------------
.. _preprocessing_api:
.. currentmodule:: pyriemann.preprocessing

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Whitening

Channel selection
------------------
.. _channelselection_api:
.. currentmodule:: pyriemann.channelselection

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ElectrodeSelection
    FlatChannelRemover

Transfer Learning
-----------------
.. _transfer_api:
.. currentmodule:: pyriemann.transfer

.. autosummary::
    :toctree: generated/

    encode_domains
    decode_domains

    :template: class.rst

    TLSplitter
    TLEstimator
    TLClassifier
    TLRegressor
    TLDummy
    TLCenter
    TLScale
    TLRotate
    MDWM

Stats
------------------
.. _stats_api:
.. currentmodule:: pyriemann.stats

.. autosummary::
    :toctree: generated/
    :template: class.rst

    PermutationDistance
    PermutationModel

Datasets
------------------
.. _datasets_api:
.. currentmodule:: pyriemann.datasets

.. autosummary::
    :toctree: generated/

    make_gaussian_blobs
    make_outliers
    make_matrices
    make_masks
    sample_gaussian_spd
    make_classification_transfer

    :template: class.rst

    RandomOverSampler

Utils function
--------------

Utils functions are low level functions that implement most base components of Riemannian geometry.

Covariance processing
~~~~~~~~~~~~~~~~~~~~~
.. _covariance_api:
.. currentmodule:: pyriemann.utils.covariance

.. autosummary::
    :toctree: generated/

    covariances
    covariance_mest
    covariance_sch
    covariance_scm
    covariances_EP
    covariances_X
    block_covariances
    cross_spectrum
    cospectrum
    coherence
    normalize
    get_nondiag_weight

Distances
~~~~~~~~~~~~~~~~~~~~~~
.. _distance_api:
.. currentmodule:: pyriemann.utils.distance

.. autosummary::
    :toctree: generated/

    distance
    distance_chol
    distance_euclid
    distance_harmonic
    distance_kullback
    distance_kullback_sym
    distance_logchol
    distance_logdet
    distance_logeuclid
    distance_poweuclid
    distance_riemann
    distance_thompson
    distance_wasserstein
    pairwise_distance

    distance_mahalanobis

Means
~~~~~~~~~~~~~~~~~~~~~~
.. _mean_api:
.. currentmodule:: pyriemann.utils.mean

.. autosummary::
    :toctree: generated/

    mean_covariance
    mean_ale
    mean_alm
    mean_chol
    mean_euclid
    mean_harmonic
    mean_kullback_sym
    mean_logchol
    mean_logdet
    mean_logeuclid
    mean_power
    mean_poweuclid
    mean_riemann
    mean_thompson
    mean_wasserstein
    maskedmean_riemann
    nanmean_riemann

Medians
~~~~~~~~~~~~~~~~~~~~~~
.. _median_api:
.. currentmodule:: pyriemann.utils

.. autosummary::
    :toctree: generated/

    median_euclid
    median_riemann

Geodesics
~~~~~~~~~~~~~~~~~~~~~~
.. _geodesic_api:
.. currentmodule:: pyriemann.utils.geodesic

.. autosummary::
    :toctree: generated/

    geodesic
    geodesic_chol
    geodesic_euclid
    geodesic_logchol
    geodesic_logeuclid
    geodesic_riemann
    geodesic_thompson
    geodesic_wasserstein

Kernels
~~~~~~~~~~~~~~~~~~~~~~
.. _kernel_api:
.. currentmodule:: pyriemann.utils.kernel

.. autosummary::
    :toctree: generated/

    kernel
    kernel_euclid
    kernel_logeuclid
    kernel_riemann

Tangent Space
~~~~~~~~~~~~~~~~~~~~~~
.. _ts_base_api:
.. currentmodule:: pyriemann.utils.tangentspace

.. autosummary::
    :toctree: generated/

    exp_map
    exp_map_euclid
    exp_map_logchol
    exp_map_logeuclid
    exp_map_riemann
    exp_map_wasserstein
    log_map
    log_map_euclid
    log_map_logchol
    log_map_logeuclid
    log_map_riemann
    log_map_wasserstein
    upper
    unupper
    tangent_space
    untangent_space
    transport
    transport_euclid
    transport_logchol
    transport_logeuclid
    transport_riemann

Base
~~~~~~~~~~~~~~~~~~~~~~
.. _base_api:
.. currentmodule:: pyriemann.utils.base

.. autosummary::
    :toctree: generated/

    ctranspose
    expm
    invsqrtm
    logm
    powm
    sqrtm
    nearest_sym_pos_def
    ddexpm
    ddlogm

Aproximate Joint Diagonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _ajd_api:
.. currentmodule:: pyriemann.utils.ajd

.. autosummary::
    :toctree: generated/

    ajd
    ajd_pham
    rjd
    uwedge

Matrix Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _mat_test_api:
.. currentmodule:: pyriemann.utils.test

.. autosummary::
    :toctree: generated/

    is_square
    is_sym
    is_skew_sym
    is_real
    is_real_type
    is_hermitian
    is_pos_def
    is_pos_semi_def
    is_sym_pos_def
    is_sym_pos_semi_def
    is_herm_pos_def
    is_herm_pos_semi_def

Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _viz_api:
.. currentmodule:: pyriemann.utils.viz

.. autosummary::
    :toctree: generated/

    plot_bihist
    plot_biscatter
    plot_cospectra
    plot_cov_ellipse
    plot_embedding
    plot_waveforms
