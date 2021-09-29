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
    :template: class.rst

    Embedding

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
.. currentmodule:: pyriemann.datasets.simulated

.. autosummary::
    :toctree: generated/
    :template: class.rst

    make_gaussian_blobs
    make_outliers    

Sampling
------------------
.. _sampling_api:
.. currentmodule:: pyriemann.datasets.sampling

.. autosummary::
    :toctree: generated/
    :template: class.rst

    sample_gaussian_spd
    generate_random_spd_matrix      


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
    distance_riemann
    distance_logeuclid
    distance_logdet
    distance_kullback
    distance_kullback_sym
    distance_wasserstein


Mean
~~~~~~~~~~~~~~~~~~~~~~
.. _mean_api:
.. currentmodule:: pyriemann.utils.mean

.. autosummary::
    :toctree: generated/

    mean_covariance
    mean_euclid
    mean_riemann
    mean_logeuclid
    mean_logdet
    mean_wasserstein
    mean_ale
    mean_alm
    mean_harmonic
    mean_kullback_sym


Geodesic
~~~~~~~~~~~~~~~~~~~~~~
.. _geodesic_api:
.. currentmodule:: pyriemann.utils.geodesic

.. autosummary::
    :toctree: generated/

    geodesic
    geodesic_riemann
    geodesic_euclid
    geodesic_logeuclid


Tangent Space
~~~~~~~~~~~~~~~~~~~~~~
.. _ts_base_api:
.. currentmodule:: pyriemann.utils.tangentspace

.. autosummary::
    :toctree: generated/

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

Aproximate Joint Diagonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _ajd_api:
.. currentmodule:: pyriemann.utils.ajd

.. autosummary::
    :toctree: generated/

    rjd
    ajd_pham
    uwedge

