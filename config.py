#########################################################################
#
# global variables 
# 
# import config as cfg in all related modules, and use a global variable
# x defined here in the other modules as cfg.x 

# Arthur Fangzhou Jiang 2017 Hebrew University

#########################################################################

import cosmo as co

import numpy as np
from scipy.interpolate import interp1d

########################## user control #################################

#---cosmology 
h = 0.7
Om = 0.3
Ob = 0.0465
OL = 0.7
s8 = 0.8
ns = 1.

#---for merger tree (the parameters for the Parkinson+08 algorithm)
M0 = 1e12 
z0 = 0.
zmax = 20.
G0 = 0.6353 
gamma1 = 0.1761
gamma2 = 0.0411

#---for satellite evolution 
Mres = 1e4 # [Msun] mass resolution
Rres = 0.001 # [kpc] spatial resolution

############################# constants #################################

G = 4.4985e-06 # gravitational constant [kpc^3 Gyr^-2 Msun^-1]
rhoc0 = 277.5 # [h^2 Msun kpc^-3]

ln10 = np.log(10.)
Root2 = np.sqrt(2.)
RootPi = np.sqrt(np.pi)
Root2OverPi = np.sqrt(2./np.pi)
Root1Over2Pi = np.sqrt(0.5/np.pi)
TwoOverRootPi = 2./np.sqrt(np.pi)
FourPiOverThree = 4.*np.pi/3.
TwoPi = 2.*np.pi
TwoPisqr = 2.*np.pi**2
ThreePi = 3.*np.pi
FourPi = 4.*np.pi
FourPiG = 4.*np.pi*G
FourPiGsqr = 4.*np.pi * G**2. # useful for dynamical friction 
ThreePiOverSixteenG = 3.*np.pi / (16.*G) # useful for dynamical time

kms2kpcGyr = 1.0227 # multiplier for converting velocity from [km/s] to 
    #[kpc/Gyr] <<< maybe useless, as we may only work with kpc and Gyr 

eps = 0.001 # an infinitesimal for various purposes: e.g., if the 
    # fractional difference of a quantify between two consecutive steps
    # is smaller than cfg.eps, jump out of a loop; and e.g., for 
    # computing derivatives
    
###################### other global variables ###########################

#---for merger trees
cosmo = {
    # -- keywords for using the CosmoloPy library -- 
    'omega_M_0' : Om, 
    'omega_lambda_0' : OL, 
    'omega_b_0' : Ob, 
    'h' : h,
    'n' : ns,
    'sigma_8' : s8,
    'N_nu' : 0,
    'omega_n_0' : 0.0,
    'omega_k_0' : 0.0, 
    'baryonic_effects': False,   # True or False
    # -- user keywords -- 
    #'m_WDM' : 1.5,               # [keV], invoke WDM power spectrum
    'MassVarianceChoice': 0,     # how to compute sigma(M,z=0): 
                                 # 0=integration, 1=interpolation 
    }

print('>>> Normalizing primordial power spectrum P(k)=(k/k_0)^n_s ...')
cosmo['k0'] = co.k0(**cosmo)
print('    such that sigma(R=8Mpc/h) = %8.4f.'%(co.sigmaR(8.,**cosmo)))

print('>>> Tabulating sigma(M,z=0) ...')
lgM_grid  = np.linspace(1.,17.,1000)
sigma_grid = co.sigma(10.**lgM_grid,z=0.,**cosmo)
sigmalgM_interp = interp1d(lgM_grid, sigma_grid, kind='linear')
cosmo['MassVarianceChoice'] = 1 
print('    From now on, sigma(M,z) is computed by interpolation.')

print('>>> Tabulating z(W) and z(t_lkbk)...')
z_grid  = np.logspace(0.,2.,10000) - 1.0 # uniform in log(1+z)
W_grid = co.deltac(z_grid,Om)
zW_interp = interp1d(W_grid, z_grid, kind='linear')
tlkbk_grid = co.tlkbk(z_grid,h,Om,OL)
ztlkbk_interp = interp1d(tlkbk_grid, z_grid, kind='linear')

print('>>> Preparing output redshifts for merger trees ...')
Nmax = 500000 # maximum number of branches per tree
zsample = [z0]
dtsample = []
z = z0
while z<=zmax:
    tlkbk = co.tlkbk(z,h,Om,OL)
    tdyn = co.tdyn(z,h,Om,OL)
    dt = 0.1 * tdyn
    z = ztlkbk_interp(tlkbk+dt)
    zsample.append(z)
    dtsample.append(dt)
dtsample.append(0.) # append a zero to the end, making dtsample the same 
    # length as zsample
zsample = np.array(zsample)
dtsample = np.array(dtsample)
Wsample = co.deltac(zsample,Om)
tlkbksample = co.tlkbk(zsample,h,Om,OL)
tsample = co.t(zsample,h,Om,OL)
Dvsample = co.DeltaBN(zsample,Om,OL)
Omsample = co.Omega(zsample,Om,OL)
Nz = len(zsample)
print('    Number of output redshifts = %4i, up to z = %5.2f'\
    %(Nz,zsample.max()))
    
print('>>> Tabulating Parkinson+08 J(u_res) ...')
ures_grid = np.logspace(-6.,6.,1000)
J_grid = co.J_vec(ures_grid)
Jures_interp = interp1d(ures_grid, J_grid, kind='linear')