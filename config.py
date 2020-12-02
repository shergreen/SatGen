#########################################################################
#
# global variables 
# 
# import config as cfg in all related modules, and use a global variable
# x defined here in the other modules as cfg.x 

# Arthur Fangzhou Jiang 2017 Hebrew University
# Sheridan Beckwith Green 2020 Yale University

#########################################################################

import cosmo as co

import numpy as np
from scipy.interpolate import interp1d, RectBivariateSpline, splrep

########################## user control #################################

#---cosmology 
h = 0.7
Om = 0.3
Ob = 0.0465
OL = 0.7
s8 = 0.8
ns = 1.

#---for merger tree (the parameters for the Parkinson+08 algorithm)
M0 = 1e12 # [Msun] [DEFAULT]: Typically changed in TreeGen_Sub
Mres = 1e8 # [Msun] [DEFAULT]: mass resolution of merger tree
           # (Mres/M0 = psi_{res})
psi_res = 10**-5 # Resolution limit of merger tree
z0 = 0. # [DEFAULT]: Typically changed in TreeGen_Sub
zmax = 20.
G0 = 0.6353 
gamma1 = 0.1761
gamma2 = 0.0411

#---for satellite evolution 
phi_res = 10**-5 # Resolution in m/m_{acc}
Rres = 0.001 # [kpc] spatial resolution (Over-written in SubEvo)
lnL_pref = 0.75 # multiplier for Coulomb logarithm (fiducial 0.75)
lnL_type = 0 # indicates using log(Mh/Ms) (instantaneous)
evo_mode = 'arbres' # or 'withering'

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
    tdyn = co.tdyn(z,h,Om,OL) # NOTE: This uses BN98 for Delta
    dt = min(0.06, 0.1 * tdyn)
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

# for Green and van den Bosch (2019) transfer function
gvdb_fp = np.array([ 3.37821658e-01, -2.21730464e-04,  1.56793984e-01,
                     1.33726984e+00,  4.47757739e-01,  2.71551083e-01, 
                    -1.98632609e-01,  1.05905814e-02, -1.11879075e+00,  
                     9.26587706e-02,  4.43963825e-01, -3.46205146e-02,
                    -3.37271922e-01, -9.91000445e-02,  4.14500861e-01])

# for computing enclosed mass within Green and van den Bosch (2019)
print('>>> Building interpolation grid for Green+19 M(<r|f_b,c)...')
print('>>> Building interpolation grid for Green+19 sigma(r|f_b,c)...')
gvdb_mm = np.load('etc/gvdb_mm.npy')
gvdb_sm = np.load('etc/gvdb_sm.npy')
nfb = 100
nr = 131
ncs = 30
fb_vals_int = np.logspace(-5, 0, nfb)
# NOTE: This approach implicitly assumes that DASH concentrations correspond
# to virial concentrations, and hence that DASH truncates at the BN98 virial
# radius.
r_vals_int = np.logspace(-5.5, 1., nr)
cs_vals_int = np.logspace(0, np.log10(40), ncs)
fbv_min = np.min(fb_vals_int) # Same as phi_{res} in paper; fiducial of 10^-5
assert phi_res >= fbv_min, "phi_res can't be smaller than fbv_min=10^-5"
fbv_max = np.max(fb_vals_int)
rv_min = np.min(r_vals_int)
rv_max = np.max(r_vals_int)
csv_min = np.min(cs_vals_int)
csv_max = np.max(cs_vals_int)
log_fb_vals_int = np.log10(fb_vals_int)
log_r_vals_int = np.log10(r_vals_int)
log_cs_vals_int = np.log10(cs_vals_int)
fb_cs_interps_mass = []
fb_cs_interps_sigma = []
# TODO: Decide if switching to linear-space from log-space gives
# a speed-up sufficiently worth it..?
for i in range(0, nr):
    fb_cs_interps_mass.append(RectBivariateSpline(log_fb_vals_int,
                                                  log_cs_vals_int,
                                                  gvdb_mm[:,:,i]))
    fb_cs_interps_sigma.append(RectBivariateSpline(log_fb_vals_int,
                                                   log_cs_vals_int,
                                                   gvdb_sm[:,:,i]))

# Jiang+15 subhalo orbital model parameters (Table 2)
# rows correspond to host mass (i.e., peak height)
# columns correspond to msub/mhost
print('>>> Building interpolator for Jiang+15 orbit sampler...')
ncdf_pts = 100
V_by_V200c_arr = np.linspace(0., 2.6, ncdf_pts)
Vr_by_V_arr = np.linspace(0., 1., ncdf_pts)
jiang_cdfs = np.load('etc/jiang_cdfs.npz')
V_by_V200c_cdf = jiang_cdfs['V_by_V200c']
Vr_by_V_cdf = jiang_cdfs['Vr_by_V']

V_by_V200c_interps = []
Vr_by_V_interps = []

for i in range(0,3):
    V_by_V200c_interps.append([])
    Vr_by_V_interps.append([])
    for j in range(0,3):
        V_by_V200c_interps[i].append(splrep(V_by_V200c_cdf[i,j], 
                                            V_by_V200c_arr))
        Vr_by_V_interps[i].append(splrep(Vr_by_V_cdf[i,j],
                                         Vr_by_V_arr))

jiang_nu_boundaries = co.nu([5e12, 5e13], 0., **cosmo)
jiang_ratio_boundaries = np.array([0.005, 0.05])
