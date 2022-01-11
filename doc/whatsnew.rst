.. _whatsnew:

.. currentmodule:: pyriemann

What's new in the package
=========================

A catalog of new features, improvements, and bug-fixes in each release.

v0.2.8.dev
----------

- Correct spectral estimation in :func:`pyriemann.utils.covariance.cross_spectrum` to obtain equivalence with SciPy

- Add instantaneous, lagged and imaginary coherences in :func:`pyriemann.utils.covariance.coherence` and :class:`pyriemann.estimation.Coherences`

- Add ``partial_fit`` in :class:`pyriemann.clustering.Potato`, useful for an online update; and update example on artifact detection.

- Deprecate :func:`pyriemann.utils.viz.plot_confusion_matrix` as sklearn integrate its own version.

- Add Ando-Li-Mathias mean estimation in :func:`pyriemann.utils.mean.mean_covariance`

- Add Schaefer-Strimmer covariance estimator in :func:`pyriemann.utils.covariance.covariances`, and an example to compare estimators

- Refactor tests + fix refit of :class:`pyriemann.tangentspace.TangentSpace`

- Add :class:`pyriemann.clustering.PotatoField`, and an example on artifact detection

- Add sampling SPD matrices from a Riemannian Gaussian distribution in :func:`pyriemann.datasets.sample_gaussian_spd`

- Add new function :func:`pyriemann.datasets.make_gaussian_blobs` for generating random datasets with SPD matrices

- Add module ``pyriemann.utils.viz`` in API, add :func:`pyriemann.utils.viz.plot_waveforms`, and add an example on ERP visualization

- Add a special form covariance matrix :func:`pyriemann.utils.covariance.covariances_X`

- Add masked and NaN means with Riemannian metric: :func:`pyriemann.utils.mean.maskedmean_riemann` and :func:`pyriemann.utils.mean.nanmean_riemann`

- Add ``corr`` option in :func:`pyriemann.utils.covariance.normalize`, to normalize covariance into correlation matrices

- Add block covariance matrix: :class:`pyriemann.estimation.BlockCovariances` and :func:`pyriemann.utils.covariance.block_covariances`

- Add Riemannian Locally Linear Embedding: :class:`pyriemann.embedding.LocallyLinearEmbedding` and :func:`pyriemann.embedding.locally_linear_embedding`

- Add Riemannian Kernel Function: :func:`pyriemann.utils.kernel.kernel_riemann`


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
