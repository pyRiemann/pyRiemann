.. _installing:

Installing pyRiemann
====================

The easiest way to install a stable version of pyRiemann is through ``pip``, the Python package manager:

``pip install pyriemann``

or via ``conda``:

``conda install -c conda-forge pyriemann``

For a bleeding edge version, you can clone the source code on `github <https://github.com/pyRiemann/pyRiemann>`__ and install directly the package from source.

``pip install -e .``

The install script will install the required dependencies. If you want also to build the documentation and to run the test locally, you could install all development dependencies with

``pip install -e .[docs,tests]``

If you use a zsh shell, you need to write `pip install -e .\[docs,tests\]`. If you do not know what zsh is, you could use the above command.


Dependencies
~~~~~~~~~~~~

-  Python >= 3.10

Mandatory dependencies
^^^^^^^^^^^^^^^^^^^^^^

-  `numpy <http://www.numpy.org/>`__

-  `scipy <http://www.scipy.org/>`__

-  `scikit-learn >=0.24 <http://scikit-learn.org/>`__

-  `joblib <https://joblib.readthedocs.io/>`__

Recommended dependencies
^^^^^^^^^^^^^^^^^^^^^^^^
These dependencies are recommanded to use the plotting functions of pyRiemann or to run examples and tutorials, but they are not mandatory:

- `mne-python <http://mne-tools.github.io/>`__

-  `matplotlib <https://matplotlib.org/>`__

-  `seaborn <https://seaborn.pydata.org>`__

-  `pandas <http://pandas.pydata.org/>`__
