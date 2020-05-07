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

![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `[]:` from profiles import NFW

![#c5f015](https://via.placeholder.com/15/c5f015/000000?text=+) `[]:` h = NFW(1e12, 10, Delta=200., z=0.)

This defines a halo object "h" following NFW profile of a virial mass of
<img src="https://render.githubusercontent.com/render/math?math=M_\mathrm{vir}=10^{12}\M_\odot"> 
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
[<img src="https://render.githubusercontent.com/render/math?math=(\mathrm{kpc/Gyr})^2">], 
the circular velocity <img src="https://render.githubusercontent.com/render/math?math=V_\mathrm{circ}(r)">
[kpc/Gyr],
the 1D velocity dispersion <img src="https://render.githubusercontent.com/render/math?math=\sigma(r)">
[kpc/Gyr]
(under the assumption of isotropic velocity 
distributino), and etc, by accessing the corresponding attribute or 
method. For example:
 
[]: h.M(10.)

returns the halo mass within a radius of 10 kpc; and

[]: h.rmax

gives the radius [kpc] at which the circular velocity reaches the maximum.   

To initialize a disk potential:

[]: from profiles import MN

[]: d = MN(10**10.7, 6.5, 0.25)

This defines a disc object "d" of mass 
<img src="https://render.githubusercontent.com/render/math?math=M_{\rm d}=10^{10.7}\M_\odot"> 
with a scale radius of 
<img src="https://render.githubusercontent.com/render/math?math=a=6.5"> kpc 
and a scale height of 
<img src="https://render.githubusercontent.com/render/math?math=b=0.25"> kpc. 

Similarly, one can evaluates various quantities by accessing the 
attributes. For example,

[]: d.Phi(8.,z=0.)

returns the gravitational potential at the cylindrical coordinate 
<img src="https://render.githubusercontent.com/render/math?math=(R,z)=(8,0)">.

One can make a composite potential simply by creating a list:

[]: p = [h,d]

This creates a composite potential consisting of the NFW halo and the 
MN disk defined above. Then, the properties of the composite potential 
can be evaluated easily. For example, if we want to get the circular 
velocity profile, we do

[]: import numpy as np

[]: R = np.logspace(-3,0,100)

[]: Vcirc(p,R,z=0.)

Let's say, we now want to integrate the orbit of a point mass
<img src="https://render.githubusercontent.com/render/math?math=m"> in
this composite potential "p", we do the following. First, we 
initialize an orbit, by specifying the initial 6D phase-space 
coordinate "xv" in the cylindrical frame (a list or an numpy array), xv = 
<img src="https://render.githubusercontent.com/render/math?math=[R,\phi,z,V_R,V_\phi,V_z]"> 
[kpc, radian, kpc, kpc/Gyr, kpc/Gyr, kpc/Gyr] --

[]: import orbit as orb

[]: o = orb.orbit(xv)

This gives us an orbit object "o". Then, orbit integration can be done 
by calling the o.integrate method:

[]: o.integrate(t,p)

This integrates the orbit for time "t" [Gyr] (float or array) in the 
potential well "p", without considering dynamical friction. If instead 
we want to consider dynamical friction, we simply add the test-object's 
mass "m" as a third parameter when calling the o.integrate function, i.e., 

[]: o.integrate(t,p,m)

After the orbit integration, we can access the instantaneous 6D 
coordinate simply by

[]: o.xv

and access the time elapsed since the initial position by

[]: o.t

If the "t" used in "o.integrate(t,p,m)" is an array, the full orbit can 
be accessed by

[]:o.xvArray

Moving on to a more realistic exercise, we recommend interested readers 
to follow the program test_evolve.py, which includes an example of 
evolving a satellite galaxy in a static host potential. This example 
also considers tidal stripping, heating, and ram-pressure, and thus 
serves as a good walk-through of not only the profiles.py and orbit.py
modules, but also the galhalo.py, init.py, and evolve.py modules. 

- Advanced usage

For full cosmological usage, TreeGen.py and SatEvo.py constitute a 
complete set of exercises. TreeGen.py generates EPS merger trees and 
initializes satellites at the first virial-crossing. SatEvo.py evolves 
the satellites. 

These programs are process-based parallelized using python's 
multiprocessing library. 

SatGen has detailed docstrings, and all the example programs are designed 
to be self-explanatory. Please feel free to contact the authors 
Fangzhou Jiang (fzjiang@caltech.edu) and 
Sheridan Beckwith Green (sheridan.green@yale.edu)
if you have questions. 
