.. _api_ref:

=============
API reference
=============

Covariance Estimation
---------------------
.. _estimation_api:
.. currentmodule:: pyriemann.estimation

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Covariances
    ERPCovariances
    XdawnCovariances
    BlockCovariances
    CospCovariances
    Coherences
    HankelCovariances
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

Classification
--------------
.. _classification_api:
.. currentmodule:: pyriemann.classification

.. autosummary::
    :toctree: generated/
    :template: class.rst

    MDM
    FgMDM
    TSclassifier
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
    TLStretch
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
    make_covariances
    make_masks
    sample_gaussian_spd
    generate_random_spd_matrix
    make_classification_transfer

Utils function
--------------

Utils functions are low level functions that implement most base components of Riemannian Geometry.

Covariance preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~
.. _covariance_api:
.. currentmodule:: pyriemann.utils.covariance

.. autosummary::
    :toctree: generated/

    covariances
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
    distance_euclid
    distance_harmonic
    distance_kullback
    distance_kullback_sym
    distance_logdet
    distance_logeuclid
    distance_riemann
    distance_wasserstein

Means
~~~~~~~~~~~~~~~~~~~~~~
.. _mean_api:
.. currentmodule:: pyriemann.utils.mean

.. autosummary::
    :toctree: generated/

    mean_covariance
    mean_ale
    mean_alm
    mean_euclid
    mean_harmonic
    mean_identity
    mean_kullback_sym
    mean_logdet
    mean_logeuclid
    mean_power
    mean_riemann
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
    geodesic_euclid
    geodesic_logeuclid
    geodesic_riemann

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

    exp_map_euclid
    exp_map_logeuclid
    exp_map_riemann
    log_map_euclid
    log_map_logeuclid
    log_map_riemann
    upper
    unupper
    tangent_space
    untangent_space

Base
~~~~~~~~~~~~~~~~~~~~~~
.. _base_api:
.. currentmodule:: pyriemann.utils.base

.. autosummary::
    :toctree: generated/

    sqrtm
    invsqrtm
    expm
    logm
    powm
    nearest_sym_pos_def

Aproximate Joint Diagonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _ajd_api:
.. currentmodule:: pyriemann.utils.ajd

.. autosummary::
    :toctree: generated/

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
    is_hermitian
    is_pos_def
    is_pos_semi_def
    is_sym_pos_def
    is_sym_pos_semi_def

Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _viz_api:
.. currentmodule:: pyriemann.utils.viz

.. autosummary::
    :toctree: generated/

    plot_confusion_matrix
    plot_embedding
    plot_cospectra
    plot_waveforms
