.. _whatsnew:

.. currentmodule:: pyriemann

What's new in the package
=========================

A catalog of new features, improvements, and bug-fixes in each release.

v0.5.dev
--------

- Correct :func:`pyriemann.utils.distance.pairwise_distance` for non-symmetric metrics. :pr:`229` by :user:`qbarthelemy`

- Correct :func:`pyriemann.utils.mean.mean_covariances` used with keyword arguments. :pr:`230` by :user:`qbarthelemy`

- Add functions to test HPD and HPSD matrices, :func:`pyriemann.utils.test.is_herm_pos_def` and :func:`pyriemann.utils.test.is_herm_pos_semi_def`. :pr:`231` by :user:`qbarthelemy`

- Add function :func:`pyriemann.datasets.simulated.make_matrices` to generate SPD, SPSD, HPD and HPSD matrices.
  Deprecate function :func:`pyriemann.datasets.simulated.make_covariances`. :pr:`232` by :user:`qbarthelemy`

- Add tests for matrix operators and distances for HPD matrices, complete doc and add references. :pr:`234` by :user:`qbarthelemy`


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

- Correct spectral estimation in :func:`pyriemann.utils.covariance.cross_spectrum` to obtain equivalence with SciPy. :pr:`133` by :user:`qbarthelemy`

- Add instantaneous, lagged and imaginary coherences in :func:`pyriemann.utils.covariance.coherence` and :class:`pyriemann.estimation.Coherences`. :pr:`132` by :user:`qbarthelemy`

- Add ``partial_fit`` in :class:`pyriemann.clustering.Potato`, useful for an online update; and update example on artifact detection. :pr:`133` by :user:`qbarthelemy`

- Deprecate :func:`pyriemann.utils.viz.plot_confusion_matrix` as sklearn integrate its own version. :pr:`135` by :user:`sylvchev`

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

- Add example on SSVEP classification

- Fix compatibility with scikit-learn v0.24

- Correct probas of :class:`pyriemann.classification.MDM`

- Add ``predict_proba`` for :class:`pyriemann.clustering.Potato`, and an example on artifact detection

- Add weights to Pham's AJD algorithm :func:`pyriemann.utils.ajd.ajd_pham`

- Add :func:`pyriemann.utils.covariance.cross_spectrum`, fix :func:`pyriemann.utils.covariance.cospectrum`; :func:`pyriemann.utils.covariance.coherence` output is kept unchanged

- Add :class:`pyriemann.spatialfilters.AJDC` for BSS and gBSS, with an example on artifact correction

- Add :class:`pyriemann.preprocessing.Whitening`, with optional dimension reduction

v0.2.6 (March 2020)
-------------------

- Updated for better Scikit-Learn v0.22 support

v0.2.5 (January 2018)
---------------------

- Added BilinearFilter

- Added a permutation test for generic scikit-learn estimator

- Stats module refactoring, with distance based t-test and f-test

- Removed two way permutation test

- Added FlatChannelRemover

- Support for python 3.5 in travis

- Added Shrinkage transformer

- Added Coherences transformer

- Added Embedding class.

v0.2.4 (June 2016)
------------------

- Improved documentation

- Added TSclassifier for out-of the box tangent space classification.

- Added Wasserstein distance and mean.

- Added NearestNeighbor classifier.

- Added Softmax probabilities for MDM.

- Added CSP for covariance matrices.

- Added Approximate Joint diagonalization algorithms (JADE, PHAM, UWEDGE).

- Added ALE mean.

- Added Multiclass CSP.

- API: param name changes in `CospCovariances` to comply to Scikit-Learn.

- API: attributes name changes in most modules to comply to the Scikit-Learn naming convention.

- Added `HankelCovariances` estimation

- Added `SPoC` spatial filtering

- Added Harmonic mean

- Added Kullback leibler mean

v0.2.3 (November 2015)
----------------------

- Added multiprocessing for MDM with joblib.

- Added kullback-leibler divergence.

- Added Riemannian Potato.

- Added sample_weight for mean estimation and MDM.
