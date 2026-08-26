.. _whatsnew:

.. currentmodule:: pyriemann

What's new in the package
=========================

A catalog of new features, improvements, and bug-fixes in each release.

v0.13.dev
---------

- Update pyRiemann from Python 3.10 - 3.12 to 3.11 - 3.13.
  :pr:`462` by :user:`qbarthelemy`

- Add ``"transductive"`` option for parameter ``target_domain`` in
  :class:`pyriemann.transfer.TLCenter`, so that ``transform()`` can recenter a
  set of inputs to its own recomputed mean instead of a stored per-domain
  center. This supports leave-one-subject-out (or leave-one-session-out)
  evaluation, where the test domain is never seen during ``fit()``.
  Also add ``"last"`` as a clearer spelling for the existing "recenter to the
  last fitted domain" behavior, deprecating the empty string ``""`` (which
  had the same meaning).
  :pr:`464` by :user:`gcattan` and :user:`aquemy`

- Add Wasserstein inner product for Hermitian matrices :func:`pyriemann.geometry.tangentspace.innerproduct_wasserstein`.
  :pr:`455` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.tangentspace.transport_wasserstein` for parallel
  transport with the Bures-Wasserstein metric.
  :pr:`476` by :user:`AmitSubhash`

- Deprecate ``pyriemann.geometry.ajd.rjd()`` renamed into :func:`pyriemann.geometry.ajd.jade`, and transpose outputed diagonalizer.
  :pr:`474` by :user:`gaoflow`

- Update ``array-api-compat`` from >=1.11 to >=1.14, and replace the deprecated
  ``array_api_extra.expand_dims`` by the namespace-native ``expand_dims``.
  :pr:`478` by :user:`AmitSubhash`

- Deprecate parameter ``probability`` of :class:`pyriemann.classification.SVC`.
  Add ``predict_proba()`` which now works without this parameter.
  :pr:`472` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.clustering.MeanShift` adding ``transform()``.
  :pr:`485` by :user:`qbarthelemy`

v0.12 (July 2026)
-----------------

- Deprecate ``pyriemann.utils.covariance.covariances_X()`` and ``pyriemann.utils.covariance.cospectrum()``.
  :pr:`442` by :user:`qbarthelemy`

- Add `Python Array API <https://data-apis.org/array-api/>`_ support for NumPy/PyTorch
  backend transparency in core utility modules (``base``, ``covariance``, ``distance``,
  ``mean``, ``geodesic``, ``tangentspace``, ``ajd``, ``kernel``, ``median``), enabling
  execution on both NumPy arrays and PyTorch tensors with optional GPU acceleration
  and autograd support.
  :pr:`433` by :user:`bruAristimunha`

- Move geometry modules (``ajd``, ``base``, ``covariance``, ``distance``,
  ``geodesic``, ``kernel``, ``mean``, ``median``, ``tangentspace``, ``test``)
  from ``pyriemann.utils`` to a new standalone ``pyriemann.geometry``
  subpackage. The old import paths (e.g. ``pyriemann.utils.mean``,
  ``pyriemann.utils.kernel``, ``pyriemann.utils.test``) still work as
  backward-compatibility shims but emit a ``DeprecationWarning``; rename to
  ``pyriemann.geometry.<module>``. The private ``pyriemann.utils._backend``
  and ``pyriemann._helpers`` modules also moved into ``pyriemann.geometry``
  so that the subpackage is fully standalone (no internal pyriemann
  imports outside of itself). Module ``pyriemann.utils.utils`` is renamed to
  ``pyriemann.utils._check`` with the same shim+warning. Tests for moved
  modules are renamed ``test_utils_*`` → ``test_geometry_*``.
  :pr:`445` by :user:`bruAristimunha`

- Enhance :func:`pyriemann.geometry.geodesic.geodesic` to accept parameter ``alpha`` as an ndarray of shape ``(...,)``,
  allowing a different geodesic position per stacked matrix pair.
  :pr:`396` by :user:`Fashad-Ahmed`

- Add example on Riemannian curvature analysis of sentence trajectories in language model embeddings,
  demonstrating how local metric tensors (SPD matrices) capture geometric structure in LLM latent spaces
  and enable classification of semantically distinct sentences using MDM.
  :pr:`448` by :user:`SzczepanK112` and :user:`gcattan`

- Add example on simulated SPD matrices to compare metrics.
  :pr:`451` by :user:`qbarthelemy`

- Add Bini-Meini-Poloni (BMP) mean :func:`pyriemann.geometry.mean.mean_bmp`,
  and Cheap mean :func:`pyriemann.geometry.mean.mean_cheap`.
  :pr:`449` by :user:`qbarthelemy`

- Move :class:`pyriemann.artifact_detection.Potato` and :class:`pyriemann.artifact_detection.PotatoField` from ``clustering`` to ``artifact_detection``.
  :pr:`453` by :user:`qbarthelemy`

- Deprecate ``pyriemann.datasets.sample_gaussian_spd()`` renamed into :func:`pyriemann.datasets.sample_gaussian`,
  and enhance it to generate HPD matrices from complex-typed mean for float ``sigma``.
  :issue:`413` by :user:`robrui`

- Add log-Cholesky inner product for Hermitian matrices :func:`pyriemann.geometry.tangentspace.innerproduct_logchol`,
  and improve other inner products.
  :pr:`450` by :user:`qbarthelemy`

- Add ``__sklearn_is_fitted__`` to stateless transformers, so they pass
  scikit-learn's ``check_is_fitted`` after ``fit()`` and can be used inside
  ``Pipeline(transform_input=...)`` (introduced in scikit-learn 1.6).
  :pr:`457` by :user:`bruAristimunha`

v0.11 (April 2026)
------------------

- Enhance ``pyriemann.datasets.sample_gaussian_spd()`` adding support for dispersion defined as a covariance matrix.
  :pr:`412` by :user:`thibaultdesurrel`

- Add :class:`pyriemann.clustering.GaussianMixture`.
  :pr:`411` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.classification.NearestConvexHull` adding ``"euclid"`` option for parameter ``metric``.
  :pr:`415` by :user:`qbarthelemy`

- Deprecate ``pyriemann.utils.mean.mean_covariance()`` renamed into :func:`pyriemann.geometry.mean.gmean`.
  :pr:`419` by :user:`qbarthelemy`

- Correct log-Euclidean parallel transport :func:`pyriemann.geometry.tangentspace.transport_logeuclid`.
  :pr:`420` by :user:`qbarthelemy`

- Add CI caching for Zenodo datasets to speed up documentation builds and avoid rate limiting.
  :pr:`417` by :user:`bruAristimunha`

- Enhance :class:`pyriemann.classification.TSClassifier` and :class:`pyriemann.transfer.TLClassifier`
  to support classifiers without parameter ``sample_weight``.
  :pr:`422` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.artifact_detection.PotatoField`, allowing a different ``metric`` per potato and adding a parameter ``method_combination``.
  :pr:`423` by :user:`qbarthelemy`

- Modernize documentation: migrate from Bootstrap to Furo theme, add card-based API navigation with sphinx-design,
  and fix Sphinx build warnings.
  :pr:`424` by :user:`bruAristimunha`

- Add ORCID identifiers and new contributors to CITATION.cff.
  :pr:`424` by :user:`bruAristimunha`

- Add functions to compute inner products.
  :pr:`428` by :user:`qbarthelemy`

- Add broadcast compatibility for utility functions, enabling batched SPD/HPD matrix operations.
  Includes broadcast-compatible Mahalanobis distance.
  :pr:`426` by :user:`bruAristimunha`

- Complete example on artifact detection by Riemannian potato field, adding a metric by potato.
  :pr:`431` by :user:`DavoudYneuro`

- Enhance :func:`pyriemann.geometry.covariance.covariance_scm` adding parameter ``weigths`` to estimate weighted sample covariance matrices.
  :pr:`434` by :user:`qbarthelemy`

v0.10 (January 2026)
--------------------

- Add example in gallery to compare clustering algorithms on synthetic datasets.
  :pr:`374` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.classification.MeanField` to be used as a feature extractor.
  :pr:`377` by :user:`qbarthelemy`

- Update pyRiemann from Python 3.9 - 3.11 to 3.10 - 3.12.
  :pr:`378` by :user:`qbarthelemy`

- Deprecate ``fit_transform()`` of :class:`pyriemann.spatialfilters.AJDC` due to incompatible dimensions.
  :pr:`382` by :user:`qbarthelemy`

- Add :class:`pyriemann.datasets.RandomOverSampler` for data augmentation of positive-definite matrices.
  :pr:`387` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.tangentspace.transport` for parallel transport of positive-definite matrices.
  :pr:`388` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.datasets.make_matrices` to generate invertible, orthogonal, unitary or non-square matrices.
  :pr:`389` by :user:`qbarthelemy`

- Deprecate ``pyriemann.utils.mean.mean_identity()``.
  :pr:`392` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.tangentspace.TangentSpace` to support HPD matrices.
  :pr:`394` by :user:`qbarthelemy`

- Speedup ALM mean :func:`pyriemann.geometry.mean.mean_alm`.
  :pr:`398` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.geodesic.geodesic_chol` and :func:`pyriemann.geometry.mean.mean_chol`.
  :pr:`399` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.tangentspace.transport_logchol` for parallel transport with log-Cholesky metric.
  :pr:`400` by :user:`qbarthelemy`

- Add Thompson metric: :func:`pyriemann.geometry.distance.distance_thompson`, :func:`pyriemann.geometry.geodesic.geodesic_thompson`,
  and :func:`pyriemann.geometry.mean.mean_thompson`.
  :pr:`401` by :user:`qbarthelemy`

- Fix sklearn error ``This Pipeline instance is not fitted yet``.
  :pr:`406` by :user:`qbarthelemy`

- Add :class:`pyriemann.classification.NearestConvexHull`.
  :pr:`405` by :user:`qbarthelemy`

v0.9 (July 2025)
----------------

- Add conjugate transpose operator :func:`pyriemann.geometry.base.ctranspose` for real- and complex-valued ndarrays.
  :pr:`348` by :user:`qbarthelemy`

- Add :class:`pyriemann.embedding.TSNE`, a Riemannian t-SNE implementation, and update example comparing embeddings.
  :pr:`347` by :user:`thibaultdesurrel`

- Fix matplotlib warning.
  :pr:`351` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.geometry.mean.mean_power` using ``init``, ``tol`` and ``maxiter`` parameters when ``p=0``.
  :pr:`353` by :user:`toncho11`

- Enhance :func:`pyriemann.geometry.distance.pairwise_distance` to support HPD matrices for ``"euclid"``,
  ``"harmonic"``, ``"logchol"`` and ``"logeuclid"`` metrics.
  :pr:`350` by :user:`qbarthelemy`

- Avoid duplicating code when using joblib.
  :pr:`359` by :user:`qbarthelemy`

- Fix sklearn warning ``This Pipeline instance is not fitted yet``.
  :pr:`358` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.tangentspace.log_map` and :func:`pyriemann.geometry.tangentspace.exp_map`.
  :pr:`363` by :user:`qbarthelemy`

- Add :class:`pyriemann.clustering.MeanShift`, a Riemannian mean shift clustering algorithm.
  :pr:`364` by :user:`qbarthelemy`

- Fix :class:`pyriemann.spatialfilters.Xdawn` removing parameter ``sample_weight`` that is not used on the fit_transformation.
  :pr:`371` by :user:`bruAristimunha`

v0.8 (February 2025)
--------------------

- Enhance :func:`pyriemann.geometry.mean.mean_ale` adding parameter ``init``,
  and add function ``check_init()`` useful to all ajd and mean functions allowing initialization.
  :pr:`328` by :user:`qbarthelemy`

- Enhance ``pyriemann.utils.mean.mean_covariance()`` to support ``"power"`` and ``"poweuclid"`` metrics.
  :pr:`329` by :user:`qbarthelemy`

- Add tangent space alignment (TSA) in transfer learning module.
  Deprecate ``pyriemann.transfer.TLStretch`` renamed into :class:`pyriemann.transfer.TLScale`,
  and ``pyriemann.classification.TSclassifier`` into :class:`pyriemann.classification.TSClassifier`.
  :pr:`320` by :user:`qbarthelemy`

- Add an example using fNIRS data with a new estimator called ``HybridBlocks`` for classifying HbO and HbR signals.
  :pr:`323` by :user:`timnaher`

- Add directional derivatives :func:`pyriemann.geometry.base.ddexpm` and :func:`pyriemann.geometry.base.ddlogm`,
  and correct :func:`pyriemann.geometry.tangentspace.log_map_logeuclid` and :func:`pyriemann.geometry.tangentspace.exp_map_logeuclid`.
  :pr:`332` by :user:`gabelstein`

- Add :func:`pyriemann.geometry.tangentspace.exp_map_wasserstein`, :func:`pyriemann.geometry.tangentspace.log_map_wasserstein`
  and :func:`pyriemann.geometry.geodesic.geodesic_wasserstein`.
  :pr:`331` by :user:`gabelstein`

- Enhance :class:`pyriemann.classification.KNearestNeighbor`, :class:`pyriemann.classification.MeanField`, :func:`pyriemann.classification.class_distinctiveness`,
  :class:`pyriemann.artifact_detection.Potato`, :class:`pyriemann.artifact_detection.PotatoField`
  :class:`pyriemann.clustering.Kmeans`, :class:`pyriemann.clustering.KmeansPerClassTransform`,
  :class:`pyriemann.embedding.SpectralEmbedding`, and :class:`pyriemann.regression.KNearestNeighborRegressor`
  to process HPD matrices.
  :pr:`336` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.datasets.make_matrices`, to generate symmetric and Hermitian matrices,
  and add parameters defining the normal distribution to draw eigen vectors.
  Deprecate ``pyriemann.datasets.generate_random_spd_matrix()``.
  :pr:`339` by :user:`qbarthelemy`

- Enhance TSA, adding parameter ``sample_weight`` to tangent space transformers,
  and generalizing :class:`pyriemann.transfer.TLRotate` from one-to-one to many-to-one domain adaptation in tangent space.
  :pr:`337` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.geometry.kernel.kernel_euclid` and
  :func:`pyriemann.geometry.kernel.kernel_logeuclid` adding parameter ``Cref``, and
  correct :func:`pyriemann.geometry.kernel.kernel_riemann` when ``Y`` is different from ``X`` and ``Cref`` is None.
  :pr:`340` by :user:`qbarthelemy`

- Speedup Wasserstein mean :func:`pyriemann.geometry.mean.mean_wasserstein` and extend parallel transport function
  :func:`pyriemann.geometry.tangentspace.transport`.
  :pr:`341` by :user:`gabelstein`

v0.7 (October 2024)
-------------------

- Enhance :class:`pyriemann.embedding.LocallyLinearEmbedding` adding parameter ``kernel``.
  :pr:`293` by :user:`gabelstein`

- Enhance :class:`pyriemann.transfer.TLCenter`, adding possibility for parameter ``target_domain`` to be empty,
  forcing ``transform()`` to recenter matrices to the last fitted domain.
  :pr:`292` by :user:`brunaafl`

- Enhance :class:`pyriemann.classification.MDM` to process HPD matrices.
  :pr:`297` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.geometry.ajd.ajd_pham` and :func:`pyriemann.geometry.mean.mean_ale` functions to process HPD matrices.
  :pr:`299` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.preprocessing.Whitening` adding ``partial_fit()`` for online applications.
  :pr:`277` by :user:`qbarthelemy` and :user:`brentgaisford`

- Add fixture ``get_weights()`` to ``conftest.py`` and complete tests.
  :pr:`305` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.estimation.Shrinkage` to process HPD matrices.
  :pr:`307` by :user:`qbarthelemy`

- Add remote sensing examples on radar image clustering.
  :pr:`306` by :user:`AmmarMian`

- Add parameter ``sample_weight`` to ``MDM.fit_predict()``, ``Potato.fit()``, ``Potato.partial_fit()``,
  ``PotatoField.fit()``, ``PotatoField.partial_fit()`` and ``Whitening.partial_fit()``.
  :pr:`309` by :user:`qbarthelemy`

- Update pyRiemann from Python 3.8 - 3.10 to 3.9 - 3.11.
  :pr:`310` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.distance.distance_logchol` to compute log-Cholesky distance.
  :pr:`311` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.spatialfilters.CSP` adding parameter ``ajd_method``.
  :pr:`313` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.distance.distance_poweuclid` and 
  :func:`pyriemann.geometry.mean.mean_poweuclid` to use power Euclidean metric.
  :pr:`312` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.geometry.mean.mean_power`, adding parameter ``init``
  and fixing default initialization provided in the associated paper.
  :pr:`324` by :user:`toncho11`

- Add :func:`pyriemann.geometry.distance.distance_chol`, :func:`pyriemann.geometry.geodesic.geodesic_logchol`,
  :func:`pyriemann.geometry.mean.mean_logchol`, and correct :func:`pyriemann.geometry.distance.distance_logchol`.
  :pr:`322` by :user:`gabelstein`

v0.6 (April 2024)
-----------------

- Update pyRiemann from Python 3.7 - 3.9 to 3.8 - 3.10.
  :pr:`254` by :user:`qbarthelemy`

- Speedup pairwise distance function :func:`pyriemann.geometry.distance.pairwise_distance`
  by adding individual functions for ``"euclid"``, ``"harmonic"``, ``"logeuclid"`` and ``"riemann"`` metrics.
  :pr:`256` by :user:`gabelstein`

- Add :func:`pyriemann.geometry.test.is_real_type` to check the type of input arrays and
  add :func:`pyriemann.geometry.covariance.covariance_scm` allowing to process complex-valued inputs for ``"scm"`` covariance estimator.
  :pr:`251` by :user:`qbarthelemy`

- Update to Read the Docs v2.
  :pr:`260` by :user:`qbarthelemy`

- Correct :func:`pyriemann.geometry.distance.distance_wasserstein` and :func:`pyriemann.geometry.distance.distance_kullback`,
  keeping only real part.
  :pr:`267` by :user:`qbarthelemy`

- Deprecate input ``covmats`` for ``mean`` functions, renamed into ``X``.
  :pr:`252` by :user:`qbarthelemy`

- Add support for complex covariance estimation for ``"lwf"``, ``"mcd"``, ``"oas"`` and ``"sch"`` estimators.
  :pr:`274` by :user:`gabelstein`

- Deprecate input ``covtest`` for ``predict()`` of :class:`pyriemann.classification.KNearestNeighbor`, renamed into ``X``.
  :pr:`259` by :user:`qbarthelemy`

- Correct check for parameter ``kernel_fct`` of :class:`pyriemann.classification.SVC`.
  :pr:`272` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.transfer.TLCenter`, ``pyriemann.transfer.TLStretch`` and :class:`pyriemann.transfer.TLRotate` estimators
  adding parameter ``sample_weight``.
  :pr:`273` by :user:`apmellot`

- Deprecate ``pyriemann.estimation.HankelCovariances`` renamed into :class:`pyriemann.estimation.TimeDelayCovariances`.
  :pr:`275` by :user:`qbarthelemy`

- Add an example on augmented covariance matrix.
  :pr:`276` by :user:`carraraig`

- Remove function ``pyriemann.datasets.make_covariances()``.
  :pr:`280` by :user:`qbarthelemy`

- Speedup :class:`pyriemann.estimation.TimeDelayCovariances`.
  :pr:`281` by :user:`qbarthelemy`

- Enhance ``ajd`` module and add a generic function :func:`pyriemann.geometry.ajd.ajd`.
  :pr:`238` by :user:`qbarthelemy`

- Add :func:`pyriemann.utils.viz.plot_bihist`, :func:`pyriemann.utils.viz.plot_biscatter`
  and :func:`pyriemann.utils.viz.plot_cov_ellipse` for display.
  :pr:`287` by :user:`qbarthelemy` and :user:`gcattan`

- Add :class:`pyriemann.estimation.CrossSpectra`, and 
  deprecate ``pyriemann.estimation.CospCovariances`` renamed into :class:`pyriemann.estimation.CoSpectra`.
  :pr:`288` by :user:`qbarthelemy`

v0.5 (Jun 2023)
---------------

- Fix :func:`pyriemann.geometry.distance.pairwise_distance` for non-symmetric metrics.
  :pr:`229` by :user:`qbarthelemy`

- Fix ``pyriemann.utils.mean.mean_covariance()`` used with keyword arguments.
  :pr:`230` by :user:`qbarthelemy`

- Add functions to test HPD and HPSD matrices, :func:`pyriemann.geometry.test.is_herm_pos_def` and
  :func:`pyriemann.geometry.test.is_herm_pos_semi_def`.
  :pr:`231` by :user:`qbarthelemy`

- Deprecate function ``pyriemann.datasets.make_covariances()``,
  replaced by new function :func:`pyriemann.datasets.make_matrices` to generate SPD, SPSD, HPD and HPSD matrices.
  :pr:`232` by :user:`qbarthelemy`

- Add tests for matrix operators and distances for HPD matrices, complete doc and add references.
  :pr:`234` by :user:`qbarthelemy`

- Enhance module ``tangentspace`` to process HPD matrices.
  :pr:`236` by :user:`qbarthelemy`

- Fix regression introduced in :class:`pyriemann.spatialfilters.Xdawn` by :pr:`214`.
  :pr:`242` by :user:`qbarthelemy`

- Fix :func:`pyriemann.geometry.kernel.kernel_euclid` applied on non-symmetric matrices.
  :pr:`245` by :user:`qbarthelemy`

- Enhance all distances adding parameter ``squared``.
  :pr:`246` by :user:`qbarthelemy`

- Correct ``transform()`` and ``predict_proba()`` of :class:`pyriemann.classification.MeanField`.
  :pr:`247` by :user:`qbarthelemy`

- Enhance module ``mean`` to process HPD matrices.
  :pr:`243` by :user:`qbarthelemy`

- Correct :func:`pyriemann.geometry.distance.distance_mahalanobis` keeping only real part.
  :pr:`249` by :user:`qbarthelemy`

- Fix ``pyriemann.datasets.sample_gaussian_spd()`` used with ``sampling_method="rejection"`` on 2x2 matrices.
  :pr:`250` by :user:`mhurte`

v0.4 (Feb 2023)
---------------

- Add exponential and logarithmic maps for three main metrics: ``"euclid"``, ``"logeuclid"`` and ``"riemann"``.
  :func:`pyriemann.geometry.tangentspace.tangent_space` is splitted in two steps:
  (i) ``log_map_*()`` projecting SPD matrices into tangent space depending on the metric; and
  (ii) :func:`pyriemann.geometry.tangentspace.upper` taking the upper triangular part of matrices.
  Similarly, :func:`pyriemann.geometry.tangentspace.untangent_space` is splitted into
  (i) :func:`pyriemann.geometry.tangentspace.unupper` and (ii) ``exp_map_*()``.
  The different metrics for tangent space mapping can now be defined into :class:`pyriemann.tangentspace.TangentSpace`,
  then used for ``transform()`` as well as for ``inverse_transform()``.
  :pr:`195` by :user:`qbarthelemy`

- Enhance AJD, adding parameter ``init`` to :func:`pyriemann.geometry.ajd.ajd_pham` and :func:`pyriemann.geometry.ajd.jade`,
  and adding ``warm_restart`` option to parameter ``dim_red`` in :class:`pyriemann.spatialfilters.AJDC`.
  :pr:`196` by :user:`qbarthelemy`

- Enhance ``pyriemann.datasets.sample_gaussian_spd()`` adding parameter ``sampling_method``,
  with ``"rejection"`` accelerating 2x2 matrices generation.
  :pr:`198` by :user:`Artim436`

- Add geometric medians for Euclidean and Riemannian metrics: :func:`pyriemann.geometry.median.median_euclid`,
  and :func:`pyriemann.geometry.median.median_riemann`,
  and add an example in gallery to compare means and medians on synthetic datasets.
  :pr:`200` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.regression.KNearestNeighborRegressor` adding ``score()``.
  :pr:`205` by :user:`qbarthelemy`

- Add a new transfer learning module ``transfer`` and examples,
  including Riemannian Procrustes Analysis (RPA) and :class:`pyriemann.transfer.MDWM`.
  :pr:`189` by :user:`plcrodrigues`, :user:`qbarthelemy` and :user:`sylvchev`

- Add class distinctiveness function to measure the distinctiveness between classes on the manifold,
  :func:`pyriemann.classification.class_distinctiveness`, and complete an example in gallery to show how it works on synthetic datasets.
  :pr:`215` by :user:`MSYamamoto`

- Add example on ensemble learning applied to functional connectivity, and add :func:`pyriemann.geometry.base.nearest_sym_pos_def`.
  :pr:`202` by :user:`mccorsi` and :user:`sylvchev`

- Add kernel matrices representation :class:`pyriemann.estimation.Kernels` and complete example comparing estimators.
  :pr:`217` by :user:`qbarthelemy`

- Add a new covariance estimator, robust fixed point covariance, and add kwds arguments for all covariance based functions and classes.
  :pr:`220` by :user:`qbarthelemy`

- Add example in gallery on frequency band selection using class distinctiveness measure.
  :pr:`219` by :user:`MSYamamoto`

- Add :func:`pyriemann.geometry.covariance.covariance_mest` supporting three robust M-estimators (Huber, Student-t and Tyler)
  and available for all covariance based functions and classes; and add an example on robust covariance estimation for corrupted data.
  Add also :func:`pyriemann.geometry.distance.distance_mahalanobis` between between vectors and a Gaussian distribution.
  :pr:`223` by :user:`qbarthelemy`

v0.3 (July 2022)
----------------

- Correct spectral estimation in :func:`pyriemann.geometry.covariance.cross_spectrum` to obtain equivalence with SciPy.
  :pr:`131` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.geometry.covariance.coherence` and :class:`pyriemann.estimation.Coherences`,
  adding parameter ``coh`` for instantaneous, lagged and imaginary coherences.
  :pr:`132` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.artifact_detection.Potato` adding ``partial_fit()`` for online update; and update example on artifact detection.
  :pr:`133` by :user:`qbarthelemy`

- Deprecate ``pyriemann.utils.viz.plot_confusion_matrix()`` as sklearn integrate its own version.
  :pr:`135` by :user:`sylvchev`

- Add Ando-Li-Mathias (ALM) mean :func:`pyriemann.geometry.mean.mean_alm`.
  :pr:`56` by :user:`sylvchev`

- Add Schaefer-Strimmer covariance estimator :func:`pyriemann.geometry.covariance.covariances`, and an example to compare estimators.
  :pr:`59` by :user:`sylvchev`

- Refactor tests + fix refit of :class:`pyriemann.tangentspace.TangentSpace`.
  :pr:`136` by :user:`sylvchev`

- Add :class:`pyriemann.artifact_detection.PotatoField`, and an example on artifact detection.
  :pr:`142` by :user:`qbarthelemy`

- Add sampling SPD matrices from a Riemannian Gaussian distribution in ``pyriemann.datasets.sample_gaussian_spd()``.
  :pr:`140` by :user:`plcrodrigues`

- Add new function :func:`pyriemann.datasets.make_gaussian_blobs` for generating random datasets with SPD matrices.
  :pr:`140` by :user:`plcrodrigues`

- Add module ``pyriemann.utils.viz`` in API, add :func:`pyriemann.utils.viz.plot_waveforms`, and add an example on ERP visualization.
  :pr:`144` by :user:`qbarthelemy`

- Add a special form covariance matrix ``pyriemann.utils.covariance.covariances_X()``.
  :pr:`147` by :user:`qbarthelemy`

- Add masked and NaN means with Riemannian metric: :func:`pyriemann.geometry.mean.maskedmean_riemann`
  and :func:`pyriemann.geometry.mean.nanmean_riemann`.
  :pr:`149` by :user:`qbarthelemy` and :user:`sylvchev`

- Enhance :func:`pyriemann.geometry.covariance.normalize` adding ``corr`` option for parameter ``norm``,
  to normalize covariance into correlation matrices.
  :pr:`153` by :user:`qbarthelemy`

- Add block covariance matrix: :class:`pyriemann.estimation.BlockCovariances` and :func:`pyriemann.geometry.covariance.block_covariances`.
  :pr:`154` by :user:`gabelstein`

- Add Riemannian Locally Linear Embedding (LLE): :class:`pyriemann.embedding.LocallyLinearEmbedding` and :func:`pyriemann.embedding.locally_linear_embedding`.
  :pr:`159` by :user:`gabelstein`

- Add Riemannian kernel function: :func:`pyriemann.geometry.kernel.kernel_riemann`.
  :pr:`159` by :user:`gabelstein`

- Fix ``fit()`` in :class:`pyriemann.channelselection.ElectrodeSelection`.
  :pr:`166` by :user:`qbarthelemy`

- Add power mean estimation in :func:`pyriemann.geometry.mean.mean_power`.
  :pr:`170` by :user:`qbarthelemy` and :user:`plcrodrigues`

- Add example in gallery to compare classifiers on synthetic datasets.
  :pr:`175` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.classification.KNearestNeighbor`, adding ``predict_proba()`` and correcting attribute ``classes_``.
  :pr:`171` by :user:`qbarthelemy`

- Add Riemannian support-vector machine (SVM) classifier: :class:`pyriemann.classification.SVC`.
  :pr:`164` by :user:`gabelstein` and :user:`qbarthelemy`

- Add Riemannian support-vector machine (SVM) regressor: :class:`pyriemann.regression.SVR`.
  :pr:`164` by :user:`gabelstein` and :user:`qbarthelemy`

- Add k-nearest-neighbors (k-NN) regressor: :class:`pyriemann.regression.KNearestNeighborRegressor`.
  :pr:`164` by :user:`gabelstein`, :user:`qbarthelemy` and :user:`agramfort`

- Add Minimum Distance to Mean Field classifier: :class:`pyriemann.classification.MeanField`.
  :pr:`172` by :user:`qbarthelemy`  and :user:`plcrodrigues`

- Add example on principal geodesic analysis (PGA) for SSVEP classification.
  :pr:`169` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.distance.distance_harmonic`, and sort functions by their names in code, doc and tests.
  :pr:`183` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.datasets.make_gaussian_blobs` adding parameter ``n_jobs`` for multiprocessing with joblib.
  :pr:`179` by :user:`sylvchev`

- Fix dispersion when generating datasets with ``pyriemann.datasets.sample_gaussian_spd()``.
  :pr:`179` by :user:`sylvchev`

- Enhance modules ``base`` and ``distance`` to process ndarrays of SPD matrices.
  :pr:`186` and :pr:`187` by :user:`qbarthelemy`

- Enhance functions of module ``utils`` to process ndarrays of SPD matrices.
  :pr:`190` by :user:`qbarthelemy`

- Enhance means functions, with faster implementations and warning when convergence is not reached.
  :pr:`188` by :user:`qbarthelemy`

v0.2.7 (June 2021)
------------------

- Add example on SSVEP classification.
  :pr:`81` by :user:`sylvchev`

- Fix compatibility with scikit-learn v0.24

- Correct probabilities returned by ``predict_proba()`` of :class:`pyriemann.classification.MDM`.
  :pr:`100` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.artifact_detection.Potato` adding ``predict_proba()``, and add an example on artifact detection.
  :pr:`105` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.covariance.normalize` for the normalization of covariance matrices.
  :pr:`111` by :user:`qbarthelemy`

- Enhance Pham's AJD algorithm :func:`pyriemann.geometry.ajd.ajd_pham` adding parameter ``sample_weight``.
  :pr:`112` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.covariance.cross_spectrum` and fix ``pyriemann.utils.covariance.cospectrum()``;
  :func:`pyriemann.geometry.covariance.coherence` output is kept unchanged.
  :pr:`118` by :user:`qbarthelemy`

- Add :func:`pyriemann.geometry.covariance.get_nondiag_weight` to compute non-diagonality weights of matrices.
  :pr:`119` by :user:`qbarthelemy`

- Add :class:`pyriemann.spatialfilters.AJDC` for blind source separation (BSS) and group BSS (gBSS),
  with an example on artifact correction.
  :pr:`120` by :user:`qbarthelemy`

- Add :class:`pyriemann.preprocessing.Whitening`, with optional dimension reduction.
  :pr:`122` by :user:`qbarthelemy`

v0.2.6 (March 2020)
-------------------

- Enhance ``FgMDM`` adding ``predict_proba()``.

- Add parallel transport for Riemannian metric.

- Remove support for Python 2, and update code for better scikit-learn v0.22 support.
  :pr:`79` by :user:`alexandrebarachant`

v0.2.5 (January 2018)
---------------------

- Enhance ``Xdawn`` and ``XdawnCovariances`` adding parameter ``baseline_cov``.

- Add ``FlatChannelRemover`` transformer.
  :pr:`30` by :user:`kingjr`

- Add ``Coherences`` transformer.

- Enhance ``Potato`` adding parameters ``pos_label`` and ``neg_label``.

- Add ``BilinearFilter`` transformer.

- Add ``Embedding`` for spectral embedding with Laplacian eigenmaps and ``pairwise_distance``.
  :pr:`54` by :user:`plcrodrigues`

- Add a permutation test for generic scikit-learn estimator.

- Enhance stats module, with distance based t-test and f-test.

- Remove two-way permutation test.

- Add support for Python 3.5 in travis.

- Add ``Shrinkage`` transformer.
  :pr:`38` by :user:`alexandrebarachant`

v0.2.4 (June 2016)
------------------

- Improve documentation.

- Add ``TSclassifier`` for out-of the box tangent space classification.

- Add Wasserstein distance ``distance_wasserstein()`` and mean ``mean_wasserstein()``.

- Add ``KNearestNeighbor`` for k-NN classifier.

- Enhance ``MDM`` adding ``predict_proba()`` which provides softmax probabilities.

- Enhance ``XdawnCovariances`` adding parameter ``xdawn_estimator``.

- Add ``CSP`` for covariance matrices.

- Add approximate joint diagonalization (AJD) algorithms: ``rjd()``, ``ajd_pham()``, ``uwedge()``.

- Add ALE mean ``mean_ale()``.

- Enhance ``CSP`` to support multiclass classification.

- Correct param name in ``CospCovariances`` to comply to scikit-learn.

- Correct attributes name in most modules to comply to the scikit-learn naming convention.

- Add ``HankelCovariances`` estimation.

- Enhance ``CSP`` adding parameter ``log``, and add ``SPoC`` spatial filtering.

- Add harmonic mean ``mean_harmonic()`` and Kullback-Leibler mean ``mean_kullback_sym()``.

v0.2.3 (November 2015)
----------------------

- Enhance ``MDM`` adding parameter ``n_jobs`` for multiprocessing with joblib.

- Add Kullback-Leibler divergences ``distance_kullback()``, ``distance_kullback_right()``, ``distance_kullback_sym()``.

- Add Riemannian ``Potato`` for artifact detection.

- Enhance ``mean`` functions and ``MDM.fit()``, adding parameter ``sample_weight``.

- Enhance ``TangentSpace`` adding parameter ``sample_weight`` to ``fit()`` and ``fit_transform()``.

- Enhance ``FGDA`` adding parameter ``sample_weight`` to ``fit()`` and ``fit_transform()``.

- Enhance ``ElectrodeSelection`` adding parameter ``sample_weight`` to ``fit()``.

v0.2.2 (June 2015)
------------------

- Deprecate parameter ``est`` of ``covariances()`` and ``covariances_EP()``, renamed into ``estimator``.

- Enhance ``MDM`` adding the possibility to use a dictionary to define parameter ``metric``.

- Enhance ``ERPCovariances`` adding parameter ``svd``.

v0.2 (May 2015)
-----------------

- Add generic function ``distance`` with parameter ``metric`` supporting ``"riemann"``, ``"logeuclid"``, ``"euclid"`` and ``"logdet"`` options.

- Add generic function ``geodesic`` with parameter ``metric`` supporting ``"riemann"``, ``"logeuclid"`` and ``"euclid"`` options.

- Add ``Kmeans`` and ``KmeansPerClassTransform`` for clustering.

- Split module ``utils`` into submodules.

v0.1 (April 2015)
-----------------

- Add Minimum Distance to Mean classifier ``MDM``, ``TangentSpace``, and first ``utils`` (distance, mean, geodesic, covariance).

- Add ``FgMDM`` and ``FGDA`` for classification.

- Add ``ElectrodeSelection``, ``Covariances``, ``ERPCovariances``, ``XdawnCovariances``, ``CospCovariances`` and ``Xdawn``.

- Add examples for motor imagery and ERP classification.
