# Instrument Bias Correction With Machine Learning Algorithms: Application to Field-Portable Mass Spectrometry
============

Authors
--------
[Brice Loose](https://bloose.github.io)<sup>1</sup>, R.T. Short<sup>2</sup>
, and S.K. Tolder<sup>2</sup>.

1: [URI Graduate School of Oceanography](https://web.uri.edu/gso/), Narragansett, RI, USA.

2: [SRI International](https://www.sri.com/) Advanced Technology and Systems Division, SRI International, St. Petersburg, FL
 Pasadena, CA, USA.


Abstract
--------
Recent studies show that the vigorous seasonal cycle of the mixed layer modulates 
upper-ocean submesoscale turbulence. 
Here we provide model-based evidence that the seasonally-changing upper-ocean 
stratification in the Kuroshio Extension also modulates submesoscale (here 10-100 km) 
inertia-gravity waves. Summertime re-stratification weakens submesoscale 
turbulence but enhances inertia-gravity waves near the surface. Thus, 
submesoscale turbulence and inertia-gravity waves undergo vigorous out-of-phase 
seasonal cycles. These results imply a strong seasonal modulation of 
the accuracy of geostrophic velocity diagnosed from submesoscale 
sea-surface height delivered by the Surface Water and Ocean Topography (SWOT) 
satellite mission.

Status
----------
  The paper is in press. Comments, questions, and suggestions are welcome and warmly appreciated. Please email me at bloose@uri.edu.
Code
----
Coding performed in Python.  The GAM backfit algorithm was used for the iterative fit.  The LSTM RNN model was implemented using the Keras interface to Tensorflow.

Data
------


Support
-------
This work was supported by a grant from the National Science Foundation, Award # 1429940.

Acknowledgments
----------------
This research was supported by an award from the National Science Foundation Chemical and Biological Oceanography Program #1429940. We thank two anonymous reviewers for the comments and suggestions that have improved this manuscript. The GAM backfit algorithm is available at https://github.com/bloose/Python_GAM_Backfit. The supplemental contains annotated Python scripts and SWIMS example data to demonstrate application of the GAM and LSTM to bias correction.

**Thanks** to Cesar Rocha (crocha@ucsd.edu) for providing this template and example to follow.
