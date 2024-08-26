.. _introduction:

Introduction to pyRiemann
=========================

pyRiemann aims at being a generic package for multivariate data analysis
but has been designed around biosignals (like EEG, MEG or EMG) manipulation
applied to brain-computer interface (BCI),
estimating covariance matrices from multichannel time series,
and classifying them using the Riemannian geometry of SPD matrices.

For BCI applications, studied paradigms are motor imagery,
event-related potentials (ERP) and steady-state visually evoked potentials (SSVEP).
Using extended labels, API allows transfer learning between sessions or subjects.

Another application is remote sensing, estimating covariance matrices
over spatial coordinates of radar images using a sliding window,
and processing them using the Riemannian geometry of
SPD matrices for hyperspectral images,
or HPD matrices for synthetic-aperture radar (SAR) images.
