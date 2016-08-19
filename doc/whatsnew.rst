.. _whatsnew:

.. currentmodule:: pyriemann

What's new in the package
=========================

A catalog of new features, improvements, and bug-fixes in each release.

v0.2.5 (Future release)
-------------------

- Added FlatChannelRemover

- Support for python 3.4 in travis

- Added Shrinkage transformer

- Added Coherences transformer

v0.2.4 (June 2016)
-------------------

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
