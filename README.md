# pyRiemann

[![PyPI version](https://badge.fury.io/py/pyriemann.svg)](https://badge.fury.io/py/pyriemann)
[![Build Status](https://github.com/pyRiemann/pyRiemann/workflows/testing/badge.svg?branch=master&event=push)](https://github.com/pyRiemann/pyRiemann/actions)
[![codecov](https://codecov.io/gh/pyRiemann/pyRiemann/branch/master/graph/badge.svg)](https://codecov.io/gh/pyRiemann/pyRiemann)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.18982.svg)](http://dx.doi.org/10.5281/zenodo.18982)
[![Documentation Status](https://readthedocs.org/projects/pyriemann/badge/?version=latest)](http://pyriemann.readthedocs.io/en/latest/?badge=latest)
[![Downloads](https://pepy.tech/badge/pot)](https://pepy.tech/project/pyriemann)

pyRiemann is a python package for covariance matrices manipulation and classification through Riemannian geometry.

The primary target is classification of multivariate biosignals, like EEG, MEG or EMG.

This is work in progress ... stay tuned.

This code is BSD-licenced (3 clause).

## Documentation

The documentation is available on http://pyriemann.readthedocs.io/en/latest/

## Install

#### Using PyPI

```
pip install pyriemann
```
or using pip+git for the latest version of the code :

```
pip install git+https://github.com/pyRiemann/pyRiemann
```

Anaconda is not currently supported, if you want to use anaconda, you need to create a virtual environment in anaconda, activate it and use the above command to install it.

#### From sources

For the latest version, you can install the package from the sources using the setup.py script

```
python setup.py install
```

or in developer mode to be able to modify the sources.

```
python setup.py develop
```

## How to use it

Most of the functions mimic the scikit-learn API, and therefore can be directly used with sklearn. For example, for cross-validation classification of EEG signal using the MDM algorithm described in [4] , it is easy as :

```python
import pyriemann
from sklearn.model_selection import cross_val_score

# load your data
X = ... # your EEG data, in format Ntrials x Nchannels X Nsamples
y = ... # the labels

# estimate covariances matrices
cov = pyriemann.estimation.Covariances().fit_transform(X)

# cross validation
mdm = pyriemann.classification.MDM()

accuracy = cross_val_score(mdm, cov, y)

print(accuracy.mean())

```

You can also pipeline methods using sklearn Pipeline framework. For example, to classify EEG signal using a SVM classifier in the tangent space, described in [5] :

```python
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# load your data
X = ... # your EEG data, in format Ntrials x Nchannels X Nsamples
y = ... # the labels

# build your pipeline
covest = Covariances()
ts = TangentSpace()
svc = SVC(kernel='linear')

clf = make_pipeline(covest,ts,svc)
# cross validation
accuracy = cross_val_score(clf, X, y)

print(accuracy.mean())

```

**Check out the example folder for more examples !**

# Testing

If you make a modification, run the test suite before submitting a pull request

```
pytest
```

# Contribution Guidelines

The package aims at adopting the [Scikit-Learn](http://scikit-learn.org/stable/developers/contributing.html#contributing-code) and [MNE-Python](https://mne.tools/stable/install/contributing.html) conventions as much as possible. See their contribution guidelines before contributing to the repository.


# References

> [1] A. Barachant, M. Congedo ,"A Plug&Play P300 BCI Using Information Geometry", arXiv:1409.0107. [link](http://arxiv.org/abs/1409.0107)
>
> [2] M. Congedo, A. Barachant, A. Andreev ,"A New generation of Brain-Computer Interface Based on Riemannian Geometry", arXiv: 1310.8115. [link](http://arxiv.org/abs/1310.8115)
>
> [3] A. Barachant and S. Bonnet, "Channel selection procedure using riemannian distance for BCI applications," in 2011 5th International IEEE/EMBS Conference on Neural Engineering (NER), 2011, 348-351. [pdf](http://hal.archives-ouvertes.fr/docs/00/60/27/07/PDF/NER11_0016_FI.pdf)
>
> [4] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Multiclass Brain-Computer Interface Classification by Riemannian Geometry," in IEEE Transactions on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012. [pdf](http://hal.archives-ouvertes.fr/docs/00/68/13/28/PDF/Barachant_tbme_final.pdf)
>
> [5] A. Barachant, S. Bonnet, M. Congedo and C. Jutten, "Classification of covariance matrices using a Riemannian-based kernel for BCI applications", in NeuroComputing, vol. 112, p. 172-178, 2013. [pdf](http://hal.archives-ouvertes.fr/docs/00/82/04/75/PDF/BARACHANT_Neurocomputing_ForHal.pdf)
