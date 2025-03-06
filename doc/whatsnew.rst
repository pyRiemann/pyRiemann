.. _whatsnew:

.. currentmodule:: pyriemann

What's new in the package
=========================

A catalog of new features, improvements, and bug-fixes in each release.

- Add :class:`pyriemann.embedding.TSNE`, a Riemannian t-SNE implementation
  and update example comparing embeddings. :pr:`347` by :user:`thibaultdesurrel`

v0.9.dev
--------

v0.8 (February 2025)
--------------------

- Enhance :func:`pyriemann.utils.mean.mean_ale` adding ``init`` parameter,
  and add function ``check_init()`` useful to all ajd and mean functions allowing initialization. :pr:`328` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.utils.mean.mean_covariance` to support "power" and "poweuclid" metrics. :pr:`329` by :user:`qbarthelemy`

- Add tangent space alignment (TSA) in transfer learning module.
  ``TLStretch`` is deprecated and renamed into :class:`pyriemann.transfer._estimators.TLScale`,
  and ``TSclassifier`` into :class:`pyriemann.classification.TSClassifier`. :pr:`320` by :user:`qbarthelemy`

- Add an example using fNIRS data with a new estimator called ``HybridBlocks`` for classifying HbO and HbR signals. :pr:`323` by :user:`timnaher`

- Add directional derivatives :func:`pyriemann.utils.base.ddexpm` and :func:`pyriemann.utils.base.ddlogm`,
  and correct :func:`pyriemann.utils.tangentspace.log_map_logeuclid` and :func:`pyriemann.utils.tangentspace.exp_map_logeuclid`. :pr:`332` by :user:`gabelstein`

- Add :func:`pyriemann.utils.tangentspace.exp_map_wasserstein`, :func:`pyriemann.utils.tangentspace.log_map_wasserstein`
  and :func:`pyriemann.utils.geodesic.geodesic_wasserstein`. :pr:`331` by :user:`gabelstein`

- Enhance :func:`pyriemann.datasets.make_matrices`, to generate symmetric and Hermitian matrices,
  and add parameters defining the normal distribution to draw eigen vectors.
  Deprecate ``generate_random_spd_matrix``. :pr:`339` by :user:`qbarthelemy`

- Enhance TSA, adding weights to transformers, and generalizing :class:`pyriemann.transfer._estimators.TLRotate` from
  one-to-one to many-to-one domain adaptation in tangent space. :pr:`337` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.utils.kernel.kernel_euclid` and :func:`pyriemann.utils.kernel.kernel_logeuclid` adding ``Cref`` parameter,
  and correct :func:`pyriemann.utils.kernel.kernel_riemann` when ``Y`` is different from ``X`` and ``Cref`` is None. :pr:`340` by :user:`qbarthelemy`

- Speedup of the Wasserstein mean :func:`pyriemann.utils.mean.mean_wasserstein` and extension of transport function
  :func:`pyriemann.utils.tangentspace.transport`. :pr:`341` by :user:`gabelstein`

v0.7 (October 2024)
-------------------

- Add ``kernel`` parameter to :class:`pyriemann.embedding.LocallyLinearEmbedding`. :pr:`293` by :user:`gabelstein`

- Add possibility for ``target_domain`` parameter of :class:`pyriemann.transfer._estimators.TLCenter` to be empty, forcing ``transform()`` to recenter matrices to the last fitted domain. :pr:`292` by :user:`brunaafl`

- Enhance :func:`pyriemann.utils.ajd.ajd_pham` and :func:`pyriemann.utils.mean.mean_ale` functions to process HPD matrices. :pr:`299` by :user:`qbarthelemy`

- Add ``partial_fit`` function to :class:`pyriemann.preprocessing.Whitening` for online applications. :pr:`277` by :user:`qbarthelemy` and :user:`brentgaisford`

- Add ``get_weights`` fixture to conftest and complete tests. :pr:`305` by :user:`qbarthelemy`

- Enhance :class:`pyriemann.estimation.Shrinkage` to process HPD matrices. :pr:`307` by :user:`qbarthelemy`

- Add remote sensing examples on radar image clustering. :pr:`306` by :user:`AmmarMian`

- Add ``sample_weight`` parameter to ``MDM.fit_predict``, ``Potato.fit``, ``Potato.partial_fit``,
  ``PotatoField.fit``, ``PotatoField.partial_fit``, ``Whitening.partial_fit``. :pr:`309` by :user:`qbarthelemy`

- Update pyRiemann from Python 3.8 - 3.10 to 3.9 - 3.11. :pr:`310` by :user:`qbarthelemy`

- Add :func:`pyriemann.utils.distance.distance_logchol` to compute log-Cholesky distance. :pr:`311` by :user:`qbarthelemy`

- Add ``ajd_method`` parameter to :class:`pyriemann.spatialfilters.CSP`. :pr:`313` by :user:`qbarthelemy`

- Add :func:`pyriemann.utils.distance.distance_poweuclid` and :func:`pyriemann.utils.mean.mean_poweuclid` to use power Euclidean metric. :pr:`312` by :user:`qbarthelemy`

- Enhance :func:`pyriemann.utils.mean.mean_power`: add ``init`` parameter and fix default initialization provided in the associated paper. :pr:`324` by :user:`toncho11`

- Add :func:`pyriemann.utils.distance.distance_chol`, :func:`pyriemann.utils.geodesic.geodesic_logchol`, :func:`pyriemann.utils.mean.mean_logchol`,
  and correct :func:`pyriemann.utils.distance.distance_logchol`. :pr:`322` by :user:`gabelstein`

v0.6 (April 2024)
-----------------

- Update pyRiemann from Python 3.7 - 3.9 to 3.8 - 3.10. :pr:`254` by :user:`qbarthelemy`

- Speedup pairwise distance function :func:`pyriemann.utils.distance.pairwise_distance`
  by adding individual functions for 'euclid', 'harmonic', 'logeuclid' and 'riemann' metrics. :pr:`256` by :user:`gabelstein`

- Add :func:`pyriemann.utils.test.is_real_type` to check the type of input arrays and
  add :func:`pyriemann.utils.covariance.covariance_scm` allowing to process complex-valued inputs for 'scm' covariance estimator. :pr:`251` by :user:`qbarthelemy`

- Update to Read the Docs v2. :pr:`260` by :user:`qbarthelemy`

- Correct :func:`pyriemann.utils.distance.distance_wasserstein` and :func:`pyriemann.utils.distance.distance_kullback`, keeping only real part. :pr:`267` by :user:`qbarthelemy`

- Deprecate input ``covmats`` for mean functions, renamed into ``X``. :pr:`252` by :user:`qbarthelemy`

- Add support for complex covariance estimation for 'lwf', 'mcd', 'oas' and 'sch' estimators. :pr:`274` by :user:`gabelstein`

- Deprecate input ``covtest`` for predict of :class:`pyriemann.classification.KNearestNeighbor`, renamed into ``X``. :pr:`259` by :user:`qbarthelemy`

- Correct check for `kernel_fct` param of :class:`pyriemann.classification.SVC`. :pr:`272` by :user:`qbarthelemy`

- Add ``sample_weight`` parameter in TLCenter, TLStretch and TLRotate. :pr:`273` by :user:`apmellot`

- Deprecate ``HankelCovariances``, renamed into :class:`pyriemann.estimation.TimeDelayCovariances`. :pr:`275` by :user:`qbarthelemy`

- Add an example on augmented covariance matrix. :pr:`276` by :user:`carraraig`

- Remove function ``make_covariances``. :pr:`280` by :user:`qbarthelemy`

- Speedup :class:`pyriemann.estimation.TimeDelayCovariances`. :pr:`281` by :user:`qbarthelemy`

- Enhance ajd module and add a generic :func:`pyriemann.utils.ajd.ajd` function. :pr:`238` by :user:`qbarthelemy`

- Add :func:`pyriemann.utils.viz.plot_bihist`, :func:`pyriemann.utils.viz.plot_biscatter` and :func:`pyriemann.utils.viz.plot_cov_ellipse` for display. :pr:`287` 
  by :user:`qbarthelemy` and :user:`gcattan`

- Add :class:`pyriemann.estimation.CrossSpectra` and deprecate ``CospCovariances`` renamed into :class:`pyriemann.estimation.CoSpectra`. :pr:`288` by :user:`qbarthelemy`

v0.5 (Jun 2023)
---------------

- Fix :func:`pyriemann.utils.distance.pairwise_distance` for non-symmetric metrics. :pr:`229` by :user:`qbarthelemy`

- Fix :func:`pyriemann.utils.mean.mean_covariance` used with keyword arguments. :pr:`230` by :user:`qbarthelemy`

- Add functions to test HPD and HPSD matrices, :func:`pyriemann.utils.test.is_herm_pos_def` and :func:`pyriemann.utils.test.is_herm_pos_semi_def`. :pr:`231` by :user:`qbarthelemy`

- Add function :func:`pyriemann.datasets.make_matrices` to generate SPD, SPSD, HPD and HPSD matrices.
  Deprecate function ``pyriemann.datasets.make_covariances``. :pr:`232` by :user:`qbarthelemy`

- Add tests for matrix operators and distances for HPD matrices, complete doc and add references. :pr:`234` by :user:`qbarthelemy`

- Enhance tangent space module to process HPD matrices. :pr:`236` by :user:`qbarthelemy`

- Fix regression introduced in :func:`pyriemann.spatialfilters.Xdawn` by :pr:`214`. :pr:`242` by :user:`qbarthelemy`

- Fix :func:`pyriemann.utils.kernel.kernel_euclid` applied on non-symmetric matrices. :pr:`245` by :user:`qbarthelemy`

- Add argument ``squared`` to all distances. :pr:`246` by :user:`qbarthelemy`

- Correct transform and predict_proba of :class:`pyriemann.classification.MeanField`. :pr:`247` by :user:`qbarthelemy`

- Enhance mean module to process HPD matrices. :pr:`243` by :user:`qbarthelemy`

- Correct :func:`pyriemann.utils.distance.distance_mahalanobis`, keeping only real part. :pr:`249` by :user:`qbarthelemy`

- Fix :func:`pyriemann.datasets.sample_gaussian_spd` used with ``sampling_method=rejection`` on 2D matrices. :pr:`250` by :user:`mhurte`

v0.4 (Feb 2023)
---------------

- Add exponential and logarithmic maps for three main metrics: 'euclid', 'logeuclid' and 'riemann'.
  :func:`pyriemann.utils.tangentspace.tangent_space` is splitted in two steps: (i) ``log_map_*()`` projecting SPD matrices into tangent space depending on the metric; and (ii) :func:`pyriemann.utils.tangentspace.upper` taking the upper triangular part of matrices.
  Similarly, :func:`pyriemann.utils.tangentspace.untangent_space` is splitted into (i) :func:`pyriemann.utils.tangentspace.unupper` and (ii) ``exp_map_*()``.
  The different metrics for tangent space mapping can now be defined into :class:`pyriemann.tangentspace.TangentSpace`,
  then used for ``transform()`` as well as for ``inverse_transform()``. :pr:`195` by :user:`qbarthelemy`

- Enhance AJD: add ``init`` to :func:`pyriemann.utils.ajd.ajd_pham` and :func:`pyriemann.utils.ajd.rjd`,
  add ``warm_restart`` to :class:`pyriemann.spatialfilters.AJDC`. :pr:`196` by :user:`qbarthelemy`

- Add parameter ``sampling_method`` to :func:`pyriemann.datasets.sample_gaussian_spd`, with ``rejection`` accelerating 2x2 matrices generation. :pr:`198` by :user:`Artim436`

- Add geometric medians for Euclidean and Riemannian metrics: :func:`pyriemann.utils.median_euclid` and :func:`pyriemann.utils.median_riemann`,
  and add an example in gallery to compare means and medians on synthetic datasets. :pr:`200` by :user:`qbarthelemy`

- Add ``score()`` to :class:`pyriemann.regression.KNearestNeighborRegressor`. :pr:`205` by :user:`qbarthelemy`

- Add Transfer Learning module and examples, including RPA and MDWM. :pr:`189` by :user:`plcrodrigues`, :user:`qbarthelemy` and :user:`sylvchev` 

- Add class distinctiveness function to measure the distinctiveness between classes on the manifold,
  :func:`pyriemann.classification.class_distinctiveness`, and complete an example in gallery to show how it works on synthetic datasets. :pr:`215` by :user:`MSYamamoto`

- Add example on ensemble learning applied to functional connectivity, and add :func:`pyriemann.utils.base.nearest_sym_pos_def`. :pr:`202` by :user:`mccorsi` and :user:`sylvchev`

- Add kernel matrices representation :class:`pyriemann.estimation.Kernels` and complete example comparing estimators. :pr:`217` by :user:`qbarthelemy`

- Add a new covariance estimator, robust fixed point covariance, and add kwds arguments for all covariance based functions and classes. :pr:`220` by :user:`qbarthelemy`

- Add example in gallery on frequency band selection using class distinctiveness measure. :pr:`219` by :user:`MSYamamoto`

- Add :func:`pyriemann.utils.covariance.covariance_mest` supporting three robust M-estimators (Huber, Student-t and Tyler)
  and available for all covariance based functions and classes; and add an example on robust covariance estimation for corrupted data.
  Add also :func:`pyriemann.utils.distance.distance_mahalanobis` between between vectors and a Gaussian distribution. :pr:`223` by :user:`qbarthelemy`

v0.3 (July 2022)
----------------

- Correct spectral estimation in :func:`pyriemann.utils.covariance.cross_spectrum` to obtain equivalence with SciPy. :pr:`131` by :user:`qbarthelemy`

- Add instantaneous, lagged and imaginary coherences in :func:`pyriemann.utils.covariance.coherence` and :class:`pyriemann.estimation.Coherences`. :pr:`132` by :user:`qbarthelemy`

- Add ``partial_fit`` in :class:`pyriemann.clustering.Potato`, useful for an online update; and update example on artifact detection. :pr:`133` by :user:`qbarthelemy`

- Deprecate ``pyriemann.utils.viz.plot_confusion_matrix`` as sklearn integrate its own version. :pr:`135` by :user:`sylvchev`

- Add Ando-Li-Mathias mean estimation in :func:`pyriemann.utils.mean.mean_covariance`. :pr:`56` by :user:`sylvchev`

- Add Schaefer-Strimmer covariance estimator in :func:`pyriemann.utils.covariance.covariances`, and an example to compare estimators :pr:`59` by :user:`sylvchev`

- Refactor tests + fix refit of :class:`pyriemann.tangentspace.TangentSpace`. :pr:`136` by :user:`sylvchev`

- Add :class:`pyriemann.clustering.PotatoField`, and an example on artifact detection. :pr:`142` by :user:`qbarthelemy`

- Add sampling SPD matrices from a Riemannian Gaussian distribution in :func:`pyriemann.datasets.sample_gaussian_spd`. :pr:`140` by :user:`plcrodrigues`

- Add new function :func:`pyriemann.datasets.make_gaussian_blobs` for generating random datasets with SPD matrices. :pr:`140` by :user:`plcrodrigues`

- Add module ``pyriemann.utils.viz`` in API, add :func:`pyriemann.utils.viz.plot_waveforms`, and add an example on ERP visualization. :pr:`144` by :user:`qbarthelemy`

- Add a special form covariance matrix :func:`pyriemann.utils.covariance.covariances_X`. :pr:`147` by :user:`qbarthelemy`

- Add masked and NaN means with Riemannian metric: :func:`pyriemann.utils.mean.maskedmean_riemann` and :func:`pyriemann.utils.mean.nanmean_riemann`. :pr:`149` by :user:`qbarthelemy` and :user:`sylvchev`

- Add ``corr`` option in :func:`pyriemann.utils.covariance.normalize`, to normalize covariance into correlation matrices. :pr:`153` by :user:`qbarthelemy`

- Add block covariance matrix: :class:`pyriemann.estimation.BlockCovariances` and :func:`pyriemann.utils.covariance.block_covariances`. :pr:`154` by :user:`gabelstein`

- Add Riemannian Locally Linear Embedding: :class:`pyriemann.embedding.LocallyLinearEmbedding` and :func:`pyriemann.embedding.locally_linear_embedding`. :pr:`159` by :user:`gabelstein`

- Add Riemannian Kernel Function: :func:`pyriemann.utils.kernel.kernel_riemann`. :pr:`159` by :user:`gabelstein`

- Fix ``fit`` in :class:`pyriemann.channelselection.ElectrodeSelection`. :pr:`166` by :user:`qbarthelemy`

- Add power mean estimation in :func:`pyriemann.utils.mean.mean_power`. :pr:`170` by :user:`qbarthelemy` and :user:`plcrodrigues`

- Add example in gallery to compare classifiers on synthetic datasets. :pr:`175` by :user:`qbarthelemy`

- Add ``predict_proba`` in :class:`pyriemann.classification.KNearestNeighbor`, and correct attribute ``classes_``. :pr:`171` by :user:`qbarthelemy`

- Add Riemannian Support Vector Machine classifier: :class:`pyriemann.classification.SVC`. :pr:`175` by :user:`gabelstein` and :user:`qbarthelemy`

- Add Riemannian Support Vector Machine regressor: :class:`pyriemann.regression.SVR`. :pr:`175` by :user:`gabelstein` and :user:`qbarthelemy`

- Add K-Nearest-Neighbor regressor: :class:`pyriemann.regression.KNearestNeighborRegressor`. :pr:`164` by :user:`gabelstein`, :user:`qbarthelemy` and :user:`agramfort`

- Add Minimum Distance to Mean Field classifier: :class:`pyriemann.classification.MeanField`. :pr:`172` by :user:`qbarthelemy`  and :user:`plcrodrigues`

- Add example on principal geodesic analysis (PGA) for SSVEP classification. :pr:`169` by :user:`qbarthelemy`

- Add :func:`pyriemann.utils.distance.distance_harmonic`, and sort functions by their names in code, doc and tests. :pr:`183` by :user:`qbarthelemy`

- Parallelize functions for dataset generation: :func:`pyriemann.datasets.make_gaussian_blobs`. :pr:`179` by :user:`sylvchev`

- Fix dispersion when generating datasets: :func:`pyriemann.datasets.sample_gaussian_spd`. :pr:`179` by :user:`sylvchev`

- Enhance base and distance functions, to process ndarrays of SPD matrices. :pr:`186` and :pr:`187` by :user:`qbarthelemy`

- Enhance utils functions, to process ndarrays of SPD matrices. :pr:`190` by :user:`qbarthelemy`

- Enhance means functions, with faster implementations and warning when convergence is not reached. :pr:`188` by :user:`qbarthelemy`

v0.2.7 (June 2021)
------------------

- Add example on SSVEP classification. :pr:`81` by :user:`sylvchev`

- Fix compatibility with scikit-learn v0.24

- Correct probas of :class:`pyriemann.classification.MDM`. :pr:`100` by :user:`qbarthelemy`

- Add ``predict_proba`` for :class:`pyriemann.clustering.Potato`, and an example on artifact detection. :pr:`105` by :user:`qbarthelemy`

- Add weights to Pham's AJD algorithm :func:`pyriemann.utils.ajd.ajd_pham`. :pr:`112` by :user:`qbarthelemy`

- Add :func:`pyriemann.utils.covariance.cross_spectrum`, fix :func:`pyriemann.utils.covariance.cospectrum`;
  :func:`pyriemann.utils.covariance.coherence` output is kept unchanged. :pr:`118` by :user:`qbarthelemy`

- Add :class:`pyriemann.spatialfilters.AJDC` for BSS and gBSS, with an example on artifact correction. :pr:`120` by :user:`qbarthelemy`

- Add :class:`pyriemann.preprocessing.Whitening`, with optional dimension reduction. :pr:`122` by :user:`qbarthelemy`

v0.2.6 (March 2020)
-------------------

- Remove support for Python 2, and update code for better scikit-learn v0.22 support. :pr:`79` by :user:`alexandrebarachant`

v0.2.5 (January 2018)
---------------------

- Add ``BilinearFilter``.

- Add a permutation test for generic scikit-learn estimator.

- Enhance stats module, with distance based t-test and f-test.

- Remove two way permutation test.

- Add ``FlatChannelRemover``. :pr:`30` by :user:`kingjr`

- Add support for Python 3.5 in travis.

- Add ``Shrinkage`` transformer. :pr:`38` by :user:`alexandrebarachant`

- Add ``Coherences`` transformer.

- Add ``Embedding`` class. :pr:`54` by :user:`plcrodrigues`

v0.2.4 (June 2016)
------------------

- Improve documentation.

- Add ``TSclassifier`` for out-of the box tangent space classification.

- Add Wasserstein distance and mean.

- Add ``KNearestNeighbor`` classifier.

- Add softmax probabilities for ``MDM``.

- Add ``CSP`` for covariance matrices.

- Add approximate joint diagonalization algorithms: JADE, PHAM, UWEDGE.

- Add ALE mean.

- Add multiclass ``CSP``.

- Correct param name in ``CospCovariances`` to comply to scikit-learn.

- Correct attributes name in most modules to comply to the scikit-learn naming convention.

- Add ``HankelCovariances`` estimation.

- Add ``SPoC`` spatial filtering.

- Add harmonic mean.

- Add Kullback-Leibler mean.

v0.2.3 (November 2015)
----------------------

- Add multiprocessing for ``MDM`` with joblib.

- Add Kullback-Leibler divergence.

- Add Riemannian ``Potato``.

- Add sample_weight for mean estimation and ``MDM``.
