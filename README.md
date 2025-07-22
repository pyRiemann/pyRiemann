# pyRiemann

[![Code PythonVersion](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11-blue)
[![PyPI version](https://badge.fury.io/py/pyriemann.svg)](https://badge.fury.io/py/pyriemann)
[![Build Status](https://github.com/pyRiemann/pyRiemann/actions/workflows/testing.yml/badge.svg?branch=master&event=push)](https://github.com/pyRiemann/pyRiemann/actions)
[![codecov](https://codecov.io/gh/pyRiemann/pyRiemann/branch/master/graph/badge.svg)](https://codecov.io/gh/pyRiemann/pyRiemann)
[![Documentation Status](https://readthedocs.org/projects/pyriemann/badge/?version=latest)](http://pyriemann.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.593816.svg)](https://doi.org/10.5281/zenodo.593816)
[![Downloads](https://pepy.tech/badge/pot)](https://pepy.tech/project/pyriemann)

pyRiemann is a Python machine learning package based on [scikit-learn](http://scikit-learn.org/stable/modules/classes.html) API.
It provides a high-level interface for processing and classification of real (*resp*. complex)-valued multivariate data
through the Riemannian geometry of symmetric (*resp*. Hermitian)
[positive definite](https://en.wikipedia.org/wiki/Definite_matrix) (SPD) (*resp*. HPD) matrices.

The documentation is available on http://pyriemann.readthedocs.io/en/latest/

This code is BSD-licensed (3 clause).

# Description

pyRiemann aims at being a generic package for multivariate data analysis
but has been designed around [biosignals](https://en.wikipedia.org/wiki/Biosignal) (like EEG, MEG or EMG)
manipulation applied to [brain-computer interface](https://en.wikipedia.org/wiki/Brain%E2%80%93computer_interface) (BCI),
estimating [covariance matrices](https://en.wikipedia.org/wiki/Covariance_matrix) from multichannel time series,
and classifying them using the Riemannian geometry of SPD matrices [[1]](#1).

For BCI applications, studied paradigms are motor imagery [[2]](#2) [[3]](#3),
event-related potentials (ERP) [[4]](#4) and steady-state visually evoked potentials (SSVEP) [[5]](#5).
Using extended labels, API allows multisource transfer learning between sessions or subjects [[6]](#6).

Another application is [remote sensing](https://en.wikipedia.org/wiki/Remote_sensing),
estimating covariance matrices over spatial coordinates of radar images using a sliding window,
and processing them using the Riemannian geometry of
SPD matrices for [hyperspectral](https://en.wikipedia.org/wiki/Hyperspectral_imaging) images,
or HPD matrices for [synthetic-aperture radar](https://en.wikipedia.org/wiki/Synthetic-aperture_radar) (SAR) images.

# Installation

#### Using PyPI

```
pip install pyriemann
```
or using pip+git for the latest version of the code:

```
pip install git+https://github.com/pyRiemann/pyRiemann
```

#### Using conda

The package is distributed via [conda-forge](https://conda-forge.org).
You could install it in your working environment, with the following command:

```shell
conda install -c conda-forge pyriemann
```

#### From sources

For the latest version, you can install the package from the sources using ``pip``:

```shell
pip install .
```

or in editable mode to be able to modify the sources:

```shell
pip install -e .
```

# How to use

Most of the functions mimic the scikit-learn API, and therefore can be directly used with sklearn.
For example, for cross-validation classification of EEG signal using the MDM algorithm described in [[2]](#2), it is easy as:

```python
import pyriemann
from sklearn.model_selection import cross_val_score

# load your data
X = ... # EEG data, in format n_epochs x n_channels x n_times
y = ... # labels

# estimate covariance matrices
cov = pyriemann.estimation.Covariances().fit_transform(X)

# build your classifier
mdm = pyriemann.classification.MDM()

# cross validation
accuracy = cross_val_score(mdm, cov, y)

print(accuracy.mean())

```

You can also pipeline methods using sklearn pipeline framework.
For example, to classify EEG signal using a SVM classifier in the tangent space, described in [[3]](#3):

```python
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# load your data
X = ... # EEG data, in format n_epochs x n_channels x n_times
y = ... # labels

# build your pipeline
clf = make_pipeline(
    Covariances(),
    TangentSpace(),
    SVC(kernel="linear"),
)

# cross validation
accuracy = cross_val_score(clf, X, y)

print(accuracy.mean())

```

Check out the example folder for more examples.

# Contribution Guidelines

The package aims at adopting the [scikit-learn](http://scikit-learn.org/stable/developers/contributing.html#contributing-code)
and [MNE-Python](https://mne.tools/stable/install/contributing.html) conventions as much as possible.
See their contribution guidelines before contributing to the repository.

# Testing

If you make a modification, run the test suite before submitting a pull request

```
pytest
```

# How to cite

```bibtex
@software{pyriemann,
  author       = {Alexandre Barachant and
                  Quentin Barthélemy and
                  Jean-Rémi King and
                  Alexandre Gramfort and
                  Sylvain Chevallier and
                  Pedro L. C. Rodrigues and
                  Emanuele Olivetti and
                  Vladislav Goncharenko and
                  Gabriel Wagner vom Berg and
                  Ghiles Reguig and
                  Arthur Lebeurrier and
                  Erik Bjäreholt and
                  Maria Sayu Yamamoto and
                  Pierre Clisson and
                  Marie-Constance Corsi and
                  Igor Carrara and
                  Apolline Mellot and
                  Bruna Junqueira Lopes and
                  Brent Gaisford and
                  Ammar Mian and
                  Anton Andreev and
                  Gregoire Cattan and
                  Arthur Lebeurrier},
  title        = {pyRiemann},
  month        = jul,
  year         = 2025,
  version      = {v0.9},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.593816},
  url          = {https://doi.org/10.5281/zenodo.593816}
}
```

# References

<a id="1">[1]</a>
M. Congedo, A. Barachant and R. Bhatia, "Riemannian geometry for EEG-based brain-computer interfaces; a primer and a review".
Brain-Computer Interfaces, 4.3, pp. 155-174, 2017. [link](https://hal.science/hal-01570120/document)

<a id="2">[2]</a>
A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass Brain-Computer Interface Classification by Riemannian Geometry".
IEEE Transactions on Biomedical Engineering, vol. 59, no. 4, pp. 920-928, 2012. [link](https://hal.archives-ouvertes.fr/hal-00681328)

<a id="3">[3]</a>
A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of covariance matrices using a Riemannian-based kernel for BCI applications".
Neurocomputing, 112, pp. 172-178, 2013. [link](https://hal.archives-ouvertes.fr/hal-00820475/)

<a id="4">[4]</a>
A. Barachant and M. Congedo, "A Plug&Play P300 BCI Using Information Geometry".
Research report, 2014. [link](http://arxiv.org/abs/1409.0107)

<a id="5">[5]</a>
EK. Kalunga, S. Chevallier, Q. Barthélemy, K. Djouani, E. Monacelli and Y. Hamam, "Online SSVEP-based BCI using Riemannian geometry".
Neurocomputing, 191, pp. 55-68, 2014. [link](https://hal.science/hal-01351623/file/Kalunga-Chevallier-Barthelemy-Online%20SSVEP-based%20BCI%20using%20Riemannian%20Geometry-Neurocomputing-16.pdf)

<a id="6">[6]</a>
PLC. Rodrigues, C. Jutten and M. Congedo, "Riemannian Procrustes analysis: transfer learning for brain-computer interfaces".
IEEE Transactions on Biomedical Engineering, vol. 66, no. 8, pp. 2390-2401, 2018. [link](https://hal.archives-ouvertes.fr/hal-01971856)
