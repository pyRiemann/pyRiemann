.. _api_ref:

=============
API reference
=============

pyRiemann provides two levels of API: **Modules** with scikit-learn compatible
estimators and transformers, and **Utility Functions** implementing low-level
Riemannian geometry operations.

.. raw:: html

   <div class="api-section-header">
     <h2>Modules</h2>
     <p>Scikit-learn compatible estimators and transformers for BCI and SPD matrix pipelines.</p>
   </div>

   <div class="api-card-grid">

     <a class="api-card card-module" href="#estimation-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <rect x="3" y="3" width="7" height="7" fill="currentColor" opacity="0.3" stroke="currentColor"/>
             <rect x="14" y="3" width="7" height="7" fill="currentColor" opacity="0.3" stroke="currentColor"/>
             <rect x="3" y="14" width="7" height="7" fill="currentColor" opacity="0.3" stroke="currentColor"/>
             <rect x="14" y="14" width="7" height="7" fill="currentColor" opacity="0.3" stroke="currentColor"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">SPD Estimation</div>
       <div class="api-card-desc">Covariance, cross-spectra, coherence, and kernel matrix estimators.</div>
       <div class="api-card-footer">10 classes</div>
     </a>

     <a class="api-card card-module" href="#embedding-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <circle cx="5" cy="12" r="2" fill="currentColor" opacity="0.5"/>
             <circle cx="19" cy="5" r="2" fill="currentColor" opacity="0.5"/>
             <circle cx="19" cy="19" r="2" fill="currentColor" opacity="0.5"/>
             <path d="M7 12 L17 6" stroke="currentColor"/>
             <path d="M7 12 L17 18" stroke="currentColor"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Embedding</div>
       <div class="api-card-desc">Spectral embedding, locally linear embedding, and t-SNE for SPD manifolds.</div>
       <div class="api-card-footer">3 classes, 2 functions</div>
     </a>

     <a class="api-card card-module" href="#classification-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <circle cx="12" cy="12" r="9" stroke="currentColor"/>
             <circle cx="12" cy="12" r="5" stroke="currentColor"/>
             <circle cx="12" cy="12" r="1.5" fill="currentColor"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Classification</div>
       <div class="api-card-desc">MDM, FgMDM, TSClassifier, SVC, and nearest convex hull classifiers.</div>
       <div class="api-card-footer">7 classes, 1 function</div>
     </a>

     <a class="api-card card-module" href="#regression-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <line x1="3" y1="20" x2="21" y2="4" stroke="currentColor"/>
             <circle cx="6" cy="16" r="1.5" fill="currentColor" opacity="0.5"/>
             <circle cx="10" cy="14" r="1.5" fill="currentColor" opacity="0.5"/>
             <circle cx="15" cy="8" r="1.5" fill="currentColor" opacity="0.5"/>
             <circle cx="19" cy="5" r="1.5" fill="currentColor" opacity="0.5"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Regression</div>
       <div class="api-card-desc">KNN and SVR regressors operating on SPD manifolds.</div>
       <div class="api-card-footer">2 classes</div>
     </a>

     <a class="api-card card-module" href="#clustering-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <circle cx="7" cy="7" r="2" fill="currentColor" opacity="0.4"/>
             <circle cx="10" cy="9" r="1.5" fill="currentColor" opacity="0.4"/>
             <circle cx="17" cy="8" r="2" fill="currentColor" opacity="0.6"/>
             <circle cx="15" cy="10" r="1.5" fill="currentColor" opacity="0.6"/>
             <circle cx="9" cy="17" r="2" fill="currentColor" opacity="0.8"/>
             <circle cx="12" cy="16" r="1.5" fill="currentColor" opacity="0.8"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Clustering</div>
       <div class="api-card-desc">K-means, mean shift, Gaussian mixture, and Potato artifact detection.</div>
       <div class="api-card-footer">6 classes</div>
     </a>

     <a class="api-card card-module" href="#tangentspace-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <path d="M4 18 Q12 4 20 18" stroke="currentColor" fill="none"/>
             <line x1="6" y1="10" x2="18" y2="10" stroke="currentColor" stroke-dasharray="2"/>
             <circle cx="12" cy="10" r="2" fill="currentColor" opacity="0.5"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Tangent Space</div>
       <div class="api-card-desc">Projection to and from the tangent space with FGDA.</div>
       <div class="api-card-footer">2 classes</div>
     </a>

     <a class="api-card card-module" href="#spatialfilter-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <path d="M3 12 Q7 4 12 12 Q17 20 21 12" stroke="currentColor" fill="none"/>
             <path d="M3 12 Q7 20 12 12 Q17 4 21 12" stroke="currentColor" fill="none" opacity="0.3"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Spatial Filtering</div>
       <div class="api-card-desc">Xdawn, CSP, SPoC, bilinear filter, and AJDC spatial filters.</div>
       <div class="api-card-footer">5 classes</div>
     </a>

     <a class="api-card card-module" href="#preprocessing-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <path d="M4 4 L20 4 L18 20 L6 20 Z" stroke="currentColor" fill="currentColor" opacity="0.15"/>
             <line x1="8" y1="10" x2="16" y2="10" stroke="currentColor"/>
             <line x1="8" y1="14" x2="16" y2="14" stroke="currentColor"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Preprocessing</div>
       <div class="api-card-desc">Whitening transform for SPD matrices.</div>
       <div class="api-card-footer">1 class</div>
     </a>

     <a class="api-card card-module" href="#channelselection-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <rect x="3" y="5" width="18" height="14" rx="2" stroke="currentColor" fill="none"/>
             <line x1="8" y1="5" x2="8" y2="19" stroke="currentColor" opacity="0.3"/>
             <line x1="13" y1="5" x2="13" y2="19" stroke="currentColor" opacity="0.3"/>
             <rect x="9" y="5" width="4" height="14" fill="currentColor" opacity="0.15"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Channel Selection</div>
       <div class="api-card-desc">Electrode selection and flat channel removal for EEG.</div>
       <div class="api-card-footer">2 classes</div>
     </a>

     <a class="api-card card-module" href="#transfer-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <path d="M4 12 L14 12" stroke="currentColor"/>
             <path d="M11 8 L15 12 L11 16" stroke="currentColor" fill="none"/>
             <path d="M20 12 L10 12" stroke="currentColor" opacity="0.3"/>
             <path d="M13 18 L9 14" stroke="currentColor" opacity="0.3" fill="none"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Transfer Learning</div>
       <div class="api-card-desc">Domain adaptation with TLCenter, TLRotate, MDWM, and more.</div>
       <div class="api-card-footer">9 classes, 2 functions</div>
     </a>

     <a class="api-card card-module" href="#stats-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <rect x="4" y="14" width="4" height="6" fill="currentColor" opacity="0.3" stroke="currentColor"/>
             <rect x="10" y="8" width="4" height="12" fill="currentColor" opacity="0.5" stroke="currentColor"/>
             <rect x="16" y="4" width="4" height="16" fill="currentColor" opacity="0.7" stroke="currentColor"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Stats</div>
       <div class="api-card-desc">Permutation tests for distances and classification models.</div>
       <div class="api-card-footer">2 classes</div>
     </a>

     <a class="api-card card-module" href="#datasets-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <ellipse cx="12" cy="7" rx="8" ry="3" stroke="currentColor" fill="currentColor" opacity="0.15"/>
             <path d="M4 7 L4 17 C4 18.7 7.6 20 12 20 C16.4 20 20 18.7 20 17 L20 7" stroke="currentColor" fill="none"/>
             <path d="M4 12 C4 13.7 7.6 15 12 15 C16.4 15 20 13.7 20 12" stroke="currentColor" fill="none" opacity="0.4"/>
           </svg>
         </div>
         <span class="api-card-tag tag-module">Module</span>
       </div>
       <div class="api-card-title">Datasets</div>
       <div class="api-card-desc">Synthetic data generators and oversampling for SPD matrices.</div>
       <div class="api-card-footer">1 class, 6 functions</div>
     </a>

   </div>

   <div class="api-section-header">
     <h2>Utility Functions</h2>
     <p>Low-level Riemannian geometry functions for distances, means, geodesics, and matrix operations.</p>
   </div>

   <div class="api-card-grid">

     <a class="api-card card-utils" href="#covariance-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <rect x="3" y="3" width="18" height="18" rx="2" stroke="currentColor" fill="none"/>
             <line x1="3" y1="9" x2="21" y2="9" stroke="currentColor" opacity="0.3"/>
             <line x1="3" y1="15" x2="21" y2="15" stroke="currentColor" opacity="0.3"/>
             <line x1="9" y1="3" x2="9" y2="21" stroke="currentColor" opacity="0.3"/>
             <line x1="15" y1="3" x2="15" y2="21" stroke="currentColor" opacity="0.3"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Covariance</div>
       <div class="api-card-desc">Covariance estimation, cross-spectrum, normalization, and processing.</div>
       <div class="api-card-footer">11 functions</div>
     </a>

     <a class="api-card card-utils" href="#distance-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <circle cx="5" cy="12" r="2.5" stroke="currentColor" fill="currentColor" opacity="0.3"/>
             <circle cx="19" cy="12" r="2.5" stroke="currentColor" fill="currentColor" opacity="0.3"/>
             <line x1="7.5" y1="12" x2="16.5" y2="12" stroke="currentColor" stroke-dasharray="3"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Distances</div>
       <div class="api-card-desc">Riemann, LogEuclid, Wasserstein, and other SPD distance functions.</div>
       <div class="api-card-footer">14 functions</div>
     </a>

     <a class="api-card card-utils" href="#mean-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <circle cx="6" cy="8" r="1.5" fill="currentColor" opacity="0.4"/>
             <circle cx="18" cy="8" r="1.5" fill="currentColor" opacity="0.4"/>
             <circle cx="6" cy="18" r="1.5" fill="currentColor" opacity="0.4"/>
             <circle cx="18" cy="18" r="1.5" fill="currentColor" opacity="0.4"/>
             <circle cx="12" cy="13" r="3" stroke="currentColor" fill="currentColor" opacity="0.3"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Means</div>
       <div class="api-card-desc">Riemannian, Euclidean, log-Euclidean, harmonic, and other mean estimators.</div>
       <div class="api-card-footer">16 functions</div>
     </a>

     <a class="api-card card-utils" href="#median-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <line x1="12" y1="4" x2="12" y2="20" stroke="currentColor"/>
             <circle cx="6" cy="10" r="1.5" fill="currentColor" opacity="0.4"/>
             <circle cx="9" cy="14" r="1.5" fill="currentColor" opacity="0.4"/>
             <circle cx="15" cy="11" r="1.5" fill="currentColor" opacity="0.4"/>
             <circle cx="18" cy="16" r="1.5" fill="currentColor" opacity="0.4"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Medians</div>
       <div class="api-card-desc">Euclidean and Riemannian geometric median estimators.</div>
       <div class="api-card-footer">2 functions</div>
     </a>

     <a class="api-card card-utils" href="#geodesic-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <path d="M4 18 Q8 6 12 12 Q16 18 20 6" stroke="currentColor" fill="none"/>
             <circle cx="4" cy="18" r="2" fill="currentColor" opacity="0.5"/>
             <circle cx="20" cy="6" r="2" fill="currentColor" opacity="0.5"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Geodesics</div>
       <div class="api-card-desc">Shortest paths on the SPD manifold for various metrics.</div>
       <div class="api-card-footer">8 functions</div>
     </a>

     <a class="api-card card-utils" href="#kernel-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <circle cx="12" cy="12" r="8" stroke="currentColor" fill="none"/>
             <circle cx="12" cy="12" r="4" stroke="currentColor" fill="currentColor" opacity="0.15"/>
             <circle cx="12" cy="12" r="1" fill="currentColor"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Kernels</div>
       <div class="api-card-desc">Riemannian, Euclidean, and log-Euclidean kernel functions.</div>
       <div class="api-card-footer">4 functions</div>
     </a>

     <a class="api-card card-utils" href="#ts-base-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <path d="M4 18 Q12 4 20 18" stroke="currentColor" fill="none"/>
             <line x1="6" y1="10" x2="18" y2="10" stroke="currentColor" stroke-dasharray="2"/>
             <path d="M10 10 L14 10" stroke="currentColor" stroke-width="3"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Tangent Space</div>
       <div class="api-card-desc">Exponential, logarithmic maps, and parallel transport operations.</div>
       <div class="api-card-footer">19 functions</div>
     </a>

     <a class="api-card card-utils" href="#base-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <text x="4" y="17" font-size="14" font-weight="bold" fill="currentColor" font-family="serif" opacity="0.7">fx</text>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Base</div>
       <div class="api-card-desc">Matrix exponential, logarithm, square root, and power functions.</div>
       <div class="api-card-footer">9 functions</div>
     </a>

     <a class="api-card card-utils" href="#ajd-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <rect x="3" y="6" width="7" height="7" rx="1" stroke="currentColor" fill="currentColor" opacity="0.15" transform="rotate(-10 6.5 9.5)"/>
             <rect x="8" y="8" width="7" height="7" rx="1" stroke="currentColor" fill="currentColor" opacity="0.25" transform="rotate(5 11.5 11.5)"/>
             <rect x="13" y="10" width="7" height="7" rx="1" stroke="currentColor" fill="currentColor" opacity="0.35"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Approx. Joint Diag.</div>
       <div class="api-card-desc">Pham, RJD, and UWEDGE algorithms for joint diagonalization.</div>
       <div class="api-card-footer">4 functions</div>
     </a>

     <a class="api-card card-utils" href="#mat-test-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <path d="M9 12 L11 14 L15 10" stroke="currentColor" stroke-width="2.5"/>
             <rect x="4" y="4" width="16" height="16" rx="2" stroke="currentColor" fill="none"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Matrix Tests</div>
       <div class="api-card-desc">Tests for symmetry, positive definiteness, and Hermitian properties.</div>
       <div class="api-card-footer">12 functions</div>
     </a>

     <a class="api-card card-utils" href="#viz-api">
       <div class="api-card-header">
         <div class="api-card-icon">
           <svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round">
             <polyline points="4,18 8,10 12,14 16,6 20,10" stroke="currentColor" fill="none"/>
             <line x1="4" y1="20" x2="20" y2="20" stroke="currentColor" opacity="0.3"/>
             <line x1="4" y1="4" x2="4" y2="20" stroke="currentColor" opacity="0.3"/>
           </svg>
         </div>
         <span class="api-card-tag tag-utils">Utils</span>
       </div>
       <div class="api-card-title">Visualization</div>
       <div class="api-card-desc">Plotting for embeddings, cospectra, covariance ellipses, and waveforms.</div>
       <div class="api-card-footer">6 functions</div>
     </a>

   </div>

   <hr style="margin: 2.5em 0; border: none; border-top: 1px solid #e0e0e0;">

.. _estimation_api:

SPD Matrices Estimation
-----------------------
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

.. _embedding_api:

Embedding
---------
.. currentmodule:: pyriemann.embedding

.. autosummary::
    :toctree: generated/

    locally_linear_embedding
    barycenter_weights

    :template: class.rst

    SpectralEmbedding
    LocallyLinearEmbedding
    TSNE

.. _classification_api:

Classification
--------------
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
    NearestConvexHull

.. autosummary::
    :toctree: generated/
    :template: function.rst

    class_distinctiveness

.. _regression_api:

Regression
--------------
.. currentmodule:: pyriemann.regression

.. autosummary::
    :toctree: generated/
    :template: class.rst

    KNearestNeighborRegressor
    SVR

.. _clustering_api:

Clustering
------------------
.. currentmodule:: pyriemann.clustering

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Kmeans
    KmeansPerClassTransform
    MeanShift
    GaussianMixture
    Potato
    PotatoField

.. _tangentspace_api:

Tangent Space
------------------
.. currentmodule:: pyriemann.tangentspace

.. autosummary::
    :toctree: generated/
    :template: class.rst

    TangentSpace
    FGDA

.. _spatialfilter_api:

Spatial Filtering
------------------
.. currentmodule:: pyriemann.spatialfilters

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Xdawn
    CSP
    SPoC
    BilinearFilter
    AJDC

.. _preprocessing_api:

Preprocessing
-------------
.. currentmodule:: pyriemann.preprocessing

.. autosummary::
    :toctree: generated/
    :template: class.rst

    Whitening

.. _channelselection_api:

Channel selection
------------------
.. currentmodule:: pyriemann.channelselection

.. autosummary::
    :toctree: generated/
    :template: class.rst

    ElectrodeSelection
    FlatChannelRemover

.. _transfer_api:

Transfer Learning
-----------------
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

.. _stats_api:

Stats
------------------
.. currentmodule:: pyriemann.stats

.. autosummary::
    :toctree: generated/
    :template: class.rst

    PermutationDistance
    PermutationModel

.. _datasets_api:

Datasets
------------------
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

.. _covariance_api:

Covariance processing
~~~~~~~~~~~~~~~~~~~~~
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

.. _distance_api:

Distances
~~~~~~~~~~~~~~~~~~~~~~
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

.. _mean_api:

Means
~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: pyriemann.utils.mean

.. autosummary::
    :toctree: generated/

    gmean
    mean_ale
    mean_alm
    mean_chol
    mean_covariance
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

.. _median_api:

Medians
~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: pyriemann.utils

.. autosummary::
    :toctree: generated/

    median_euclid
    median_riemann

.. _geodesic_api:

Geodesics
~~~~~~~~~~~~~~~~~~~~~~
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

.. _kernel_api:

Kernels
~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: pyriemann.utils.kernel

.. autosummary::
    :toctree: generated/

    kernel
    kernel_euclid
    kernel_logeuclid
    kernel_riemann

.. _ts_base_api:

Tangent Space
~~~~~~~~~~~~~~~~~~~~~~
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

.. _base_api:

Base
~~~~~~~~~~~~~~~~~~~~~~
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

.. _ajd_api:

Aproximate Joint Diagonalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: pyriemann.utils.ajd

.. autosummary::
    :toctree: generated/

    ajd
    ajd_pham
    rjd
    uwedge

.. _mat_test_api:

Matrix Tests
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

.. _viz_api:

Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. currentmodule:: pyriemann.utils.viz

.. autosummary::
    :toctree: generated/

    plot_bihist
    plot_biscatter
    plot_cospectra
    plot_cov_ellipse
    plot_embedding
    plot_waveforms
