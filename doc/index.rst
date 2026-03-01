:hide-toc:

.. title:: pyRiemann

pyRiemann: Riemannian Geometry for Machine Learning
====================================================

Machine learning for multivariate data through the Riemannian geometry
of symmetric positive definite matrices.

.. raw:: html

   <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 1.5em 0;">
     <a href="auto_examples/biosignal-mi/plot_single.html" style="border: 1px solid #eee; border-radius: 8px; overflow: hidden; transition: box-shadow 0.3s;">
       <img src="_images/sphx_glr_plot_single_thumb.png" style="width: 100%; display: block;" alt="Motor imagery single subject">
     </a>
     <a href="auto_examples/biosignal-erp/plot_classify_MEG_mdm.html" style="border: 1px solid #eee; border-radius: 8px; overflow: hidden; transition: box-shadow 0.3s;">
       <img src="_images/sphx_glr_plot_classify_MEG_mdm_thumb.png" style="width: 100%; display: block;" alt="MEG classification MDM">
     </a>
     <a href="auto_examples/biosignal-erp/plot_classify_EEG_tangentspace.html" style="border: 1px solid #eee; border-radius: 8px; overflow: hidden; transition: box-shadow 0.3s;">
       <img src="_images/sphx_glr_plot_classify_EEG_tangentspace_thumb.png" style="width: 100%; display: block;" alt="EEG tangent space classification">
     </a>
     <a href="auto_examples/stats/plot_oneway_manova_frequency.html" style="border: 1px solid #eee; border-radius: 8px; overflow: hidden; transition: box-shadow 0.3s;">
       <img src="_images/sphx_glr_plot_oneway_manova_frequency_thumb.png" style="width: 100%; display: block;" alt="One-way MANOVA frequency">
     </a>
   </div>

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Quick Start
      :link: installing
      :link-type: doc
      :text-align: center

      Get up and running with pyRiemann.

      +++
      :ref:`Installation guide <installing>`

   .. grid-item-card:: Examples
      :link: auto_examples/index
      :link-type: doc
      :text-align: center

      Browse the example gallery for tutorials and use cases.

      +++
      `Example gallery <auto_examples/index.html>`_

   .. grid-item-card:: API Reference
      :link: api
      :link-type: doc
      :text-align: center

      Detailed documentation for all modules, classes, and functions.

      +++
      :ref:`API docs <api_ref>`

Overview
--------

pyRiemann is a Python package for machine learning with multivariate data,
using the Riemannian geometry of symmetric (resp. Hermitian) positive definite
(SPD) (resp. HPD) matrices. It provides a high-level interface fully compatible
with scikit-learn, making it easy to build powerful pipelines for
classification, regression, and clustering.

**Key features:**

- **Riemannian Geometry** -- Leverage distances, means, and tangent space
  projections on the SPD manifold.
- **scikit-learn Compatible** -- All estimators follow the scikit-learn API.
  Build pipelines, use cross-validation, and grid search out of the box.
- **Brain-Computer Interfaces** -- State-of-the-art processing and
  classification of EEG, MEG, and fNIRS signals.
- **Remote Sensing** -- Apply Riemannian methods to SAR image classification.
- **Transfer Learning** -- Domain adaptation tools for cross-session and
  cross-subject generalization.

Example
-------

Build a simple BCI classification pipeline using covariance matrices and the
Minimum Distance to Mean (MDM) classifier:

.. code-block:: python

   from pyriemann.estimation import Covariances
   from pyriemann.classification import MDM
   from sklearn.pipeline import make_pipeline

   pipeline = make_pipeline(
       Covariances(estimator="lwf"),
       MDM(metric="riemann"),
   )
   pipeline.fit(X_train, y_train)
   print(pipeline.score(X_test, y_test))

Citing pyRiemann
----------------

If you use pyRiemann in a scientific publication, please cite it using
the `Zenodo DOI <https://doi.org/10.5281/zenodo.593816>`_:

.. literalinclude:: CITATION.apa
   :language: text

**BibTeX entry:**

.. literalinclude:: CITATION.bib
   :language: bibtex

.. toctree::
   :hidden:
   :caption: Getting Started

   Introduction <introduction>
   Installing <installing>

.. toctree::
   :hidden:
   :caption: User Guide

   Example gallery <auto_examples/index>
   Release notes <whatsnew>

.. toctree::
   :hidden:
   :caption: API Documentation

   API reference <api>
