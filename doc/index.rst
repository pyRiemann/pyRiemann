.. pyRiemann documentation master file, created by
   sphinx-quickstart on Sun Apr 19 13:17:55 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. raw:: html

   <style type="text/css">
   .thumbnail {{
       position: relative;
       float: left;
       margin: 10px;
       width: 180px;
       height: 200px;
   }}

   .thumbnail img {{
       position: absolute;
       display: inline;
       left: 0;
       width: 170px;
       height: 170px;
   }}

   </style>

pyRiemann: Biosignals classification with Riemannian Geometry
=======================================

.. raw:: html


   <div style="clear: both"></div>
   <div class="container-fluid hidden-xs hidden-sm">
     <div class="row">
       <a href="auto_examples/motor-imagery/plot_single.html">
         <div class="col-md-3 thumbnail">
           <img src="_images/sphx_glr_plot_single_thumb.png">
         </div>
       </a>
       <a href="auto_examples/ERP/plot_classify_MEG_mdm.html">
         <div class="col-md-3 thumbnail">
           <img src="_images/sphx_glr_plot_classify_MEG_mdm_thumb.png">
         </div>
       </a>
       <a href="auto_examples/ERP/plot_classify_EEG_tangentspace.html">
         <div class="col-md-3 thumbnail">
           <img src="_images/sphx_glr_plot_classify_EEG_tangentspace_thumb.png">
         </div>
       </a>
       <a href="auto_examples/stats/plot_twoWay_Manova.html">
         <div class="col-md-3 thumbnail">
           <img src="_images/sphx_glr_plot_twoWay_Manova_thumb.png">
         </div>
       </a>
     </div>
   </div>
   <br>

  <div class="container-fluid">
  <div class="row">
  <div class="col-md-9">
  <br>

pyRiemann is a Python machine learning library based on scikit-learn API. It provides a high-level interface for classification and manipulation of multivariate signal through Riemannian Geometry of covariance matrices.

pyRiemann aim at being a generic package for multivariate signal classification but has been designed around applications of biosignal (M/EEG, EMG, etc) classification.

For a brief introduction to the ideas behind the package, you can read the
:ref:`introductory notes <introduction>`. More practical information is on the
:ref:`installation page <installing>`. You may also want to browse the
`example gallery <auto_examples/index.html>`_ to get a sense for what you can do with pyRiemann and then check out the :ref:`tutorial <tutorial>` and :ref:`API reference <api_ref>` to find out how.

To see the code or report a bug, please visit the `github repository
<https://github.com/alexandrebarachant/pyRiemann>`_.


.. raw:: html

  </div>
  <div class="col-md-3">
  <h2>Documentation</h2>

.. toctree::
  :maxdepth: 1

  introduction
  whatsnew
  installing
  auto_examples/index
  api
  tutorial

  </div>
  </div>
