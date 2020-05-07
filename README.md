# SatGen

A semi-analytical satellite galaxy and dark matter halo generator,
introduced in Jiang et al. (2020), extended in Green et al. (2020).

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

SatGen builds upon density-profile classes and an orbit class. Apart from
the main purpose of evolving satellites, these modules of SatGen can be
used for studies involving halo/galaxy profiles (e.g, Jeans modeling) 
and orbit integration within spherical or axisymmetric potentials. 
 
To initialize a halo, for example, we can do

[]: from profiles import NFW

[]: h = NFW(1e12, 10, Delta=200., z=0.)

This defines a halo object "h" following NFW profile of a virial mass of
<img src="https://render.githubusercontent.com/render/math?math=M_{\rm vir}=10^{12}\M_\odot"> 
and a concentration of 10, where a halo is defined as spherical enclosure 
of 200 times the critical density of the Universe at redshift 0. (Note 
that since the profiles module internally imports the cosmo module, if it 
is the first time of importing profiles, it takes a few seconds to 
initialize cosmology-related stuff.)

With the halo object defined, one can easily evaluate, at radius 
<img src="https://render.githubusercontent.com/render/math?math=r"> 
[kpc], the density <img src="https://render.githubusercontent.com/render/math?math=\rho(r)"> 
[<img src="https://render.githubusercontent.com/render/math?math=M_\odot\mathrm{kpc}^{-3}"> ], 
the enclosed mass <img src="https://render.githubusercontent.com/render/math?math=M(r)"> 
[<img src="https://render.githubusercontent.com/render/math?math=M_\odot"> ], 
the gravitational potential <img src="https://render.githubusercontent.com/render/math?math=\Phi(r)"> 
[<img src="https://render.githubusercontent.com/render/math?math=({\rm kpc/Gyr})^2">], 
the 1D velocity dispersion <img src="https://render.githubusercontent.com/render/math?math=\sigma(r)">
[kpc/Gyr]
(under the assumption of isotropic velocity 
distributino), and etc, by accessing the corresponding attribute or 
method. For example:
 
[]: h.M(10.)

gives the halo mass within a radius of 10 kpc;

[]: h.rmax

gives the radius at which the circular velocity reaches the maximum.   

To initialize a disk potential:

[]: from profiles import MN

[]: d = MN(10**10.7, 6.5, 0.25)

This defines a disc object "d" of mass 
<img src="https://render.githubusercontent.com/render/math?math=M_{\rm d}=10^{10.7}\M_\odot"> 
with a scale radius of 
<img src="https://render.githubusercontent.com/render/math?math=a=6.5{\rm kpc}"> 
and a scale height of 
<img src="https://render.githubusercontent.com/render/math?math=b=0.25{\rm kpc}">. 
