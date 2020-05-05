# SatGen

A semi-analytical satellite galaxy and dark matter halo generator.

Introduced in Jiang et al. (2020), extended in Green et al. (2020).

- Overview of the model

SatGen generates satellite-galaxy populations for halos of desired mass
and redshift. It combines halo merger trees, empirical relations for 
galaxy-halo connection, and analytic prescriptions for tidal effects, 
dynamical friction, and ram-pressure stripping. It emulates zoom-in 
cosmological hydrosimulations in some ways and outperforms simulations
regarding statistical power and numerical resolution. 

- Modules of the model

profiles.py: halo-density-profile classes, supporting NFW profile, 
Einasto profile, Dekel+ profile, MN disk

orbit.py: orbit class and equation of motion

cosmo.py: cosmology-related function, including an implementation of the
Parkinson+08 merger tree algorithm 

config.py: global variables and user controls 

galhalo.py: galaxy-halo connections

init.py: for initializing satellite properties at infall 

evolve.py: for mass stripping and structural evolution of satellites

TreeGen.py: an example program for generating halo merger trees

SatEvo.py: an example program for evolving satellite galaxies after 
generating merger trees with TreeGen.py

SatGen comes with detailed in-line documentation. 

- Dependent libraries and packages

numpy, scipy, cosmolopy

We recommend using python installations from Enthought or Conda. 

- Basic usage



