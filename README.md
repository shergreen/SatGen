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
Einasto profile, Dekel+ profile, Miyamoto-Nagai disk

orbit.py: orbit class and equation of motion

cosmo.py: cosmology-related functions, including an implementation of the
Parkinson+08 merger tree algorithm 

config.py: global variables and user controls 

galhalo.py: galaxy-halo connections

init.py: for initializing satellite properties at infall 

evolve.py: for mass stripping and structural evolution of satellites

TreeGen.py: an example program for generating halo merger trees

SatEvo.py: an example program for evolving satellite galaxies after 
generating merger trees with TreeGen.py 

- Dependent libraries and packages

numpy, scipy, cosmolopy

We recommend using python installations from Enthought or Conda. 

- Basic usage

SatGen builds upon density-profile classes and orbit class. Apart from
the main purpose of evolving satellites, these modules of SatGen can be
used for studies involving halo/galaxy profiles such as Jeans modeling, 
and doing orbit integration within spherical or axisymmetric potentials. 
 
For example, to initialize a halo:

[]: from profiles import NFW

[]: h = NFW(1e12, 10, Delta=200., z=0.)

This defines a halo "h" following NFW profile of a virial mass of
<img src="https://render.githubusercontent.com/render/math?math=10\times10^{12}\M_\odot"> and a concentration of 10, where a halo is
defined as spherical enclosure of 200 times the critical density of 
the Universe at redshift 0.
 
(Note that since the profiles module internally imports the cosmo module, 
if it is the first time importing profiles, it takes a few seconds in
cosmology related initialization.)
