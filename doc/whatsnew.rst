.. _whatsnew:

.. currentmodule:: pyriemann

What's new in the package
=========================

A catalog of new features, improvements, and bug-fixes in each release.

v0.2.8.dev
----------

- Add `partial_fit` in :class:`pyriemann.clustering.Potato`, useful for an online update; and update example on artifact detection

- Add instantaneous, lagged and imaginary coherences in :func:`pyriemann.utils.covariance.coherence` and :class:`pyriemann.estimation.Coherences`

- Correct spectral estimation in :func:`pyriemann.utils.covariance.cross_spectrum` to obtain equivalence with SciPy

v0.2.7 (June 2021)
------------------

- Add example on SSVEP classification

- Fix compatibility with scikit-learn v0.24

- Correct probas of MDM

- Add predict_proba for Potato, and an example on artifact detection

- Add weights to Pham's AJD algorithm

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
