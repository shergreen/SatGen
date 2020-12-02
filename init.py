############### Functions for initializing satellites ###################

# Arthur Fangzhou Jiang 2019, HUJI
# Sheridan Beckwith Green 2020, Yale University

#########################################################################

import numpy as np
import sys

import config as cfg
import cosmo as co
import galhalo as gh
import aux as aux

from scipy.stats import lognorm, expon
from scipy.interpolate import splrep, splev

#########################################################################

#---for initial satellite-galaxy stellar size   

def Reff(Rv,c2):
    """
    Draw half-stellar-mass radius from a Gaussian whose median is 
    given by the Jiang+19 relation, and the 1-sigma scatter is 0.12dex. 
    
    Syntax:
    
        Reff(Rv,c2)
        
    where 
    
        Rv: halo radius R_200c [kpc] (float or array)
        c2: concentration, R_00c / r_-2 (float or array of the same size
            as Rv)
        
    Note that we do not allow the half-stellar-mass radius to exceed 
    0.2 R_200c.
    
    Return:
    
        half-stellar-mass radius [kpc] (float or array of the same size
        as Rv)
    """
    mu = np.log10( gh.Reff(Rv,c2) )
    return np.minimum(0.2*Rv, 10.**np.random.normal(mu, 0.12) )

def Rvir(Mv,Delta=200.,z=0.):
    """
    Compute halo radius given halo mass, overdensity, and redshift.
    
    Syntax:
    
        Rvir(Mv,Delta=200.,z=0.)
        
    where
    
        Mv: halo mass [M_sun] (float or array)
        Delta: spherical overdensity (float, default=200.)
        z: redshift (float or array of the same size as Mv, default=0.)
    """
    rhoc = co.rhoc(z,h=cfg.h,Om=cfg.Om,OL=cfg.OL)
    return (3.*Mv / (cfg.FourPi*Delta*rhoc))**(1./3.) 

#---for initial (sub)halo profiles

# for drawing the conventional halo concentration, c_-2
def concentration(Mv,z=0.,choice='DM14'):
    """
    Draw the conventional halo concentration, c_-2, from an empirical
    concentration-mass-redshift relation of choice.
    
    Syntax:
    
        concentration(Mv,z=0.,choice='DM14')
        
    where
    
        Mv: halo mass, can be M_200c or M_vir, depending on the "choice"
            [M_sun] (float or array)
        z: redshift (float or array of the same size as Mv, default=0.)
        choice: the relation of choice (default='DM14', representing
            Dutton & Maccio 14)
            
    Note that we do not allow concentration to go below 3.
    """
    if choice=='DM14':
        mu = gh.lgc2_DM14(Mv,z)
    return np.maximum(3., 10.**np.random.normal(mu,0.1) )

# for drawing stellar mass from stellar-to-halo-mass relations (SHMR)
def Mstar(Mv,z=0.,choice='RP17'):
    """
    Stellar mass given halo mass and redshift, using abundance-matching
    relations.
    
    Syntax:
    
        Ms(Mv,z=0.,choice='RP17')
        
    where
    
        Mv: halo mass [M_sun] (float or array)
        z: instantaneous redshift (float or array of the same size as Mv,
            default=0.)
        choice: choice of the stellar-to-halo-mass relation 
            (default='RP17', representing Rodriguez-Puebla+17)
        
    Return:
    
        stellar mass [M_sun] (float or array of the same size as Mv)
    """
    if choice=='RP17':
        mu = gh.lgMs_RP17(np.log10(Mv),z)
    if choice=='B13':
        mu = gh.lgMs_B13(np.log10(Mv),z)
    return np.minimum( cfg.Ob/cfg.Om*Mv, 10.**np.random.normal(mu,0.15) )
    
# for drawing the Dekel+ parameters

def aDekel(Mv,c2,z=0.):
    """
    Draw the Dekel+ innermost slope, alpha, given halo mass, redshift, 
    and the conventional halo concentration parameter, c_-2.
     
    In particular, we use the Di Cintio + 2014 relation to compute the 
    slope, s_0.01, at r = 0.01 R_vir, assuming a 1-sigma scatter of 
    0.15; then, we express alpha with s_0.01 and c_-2.
    
    Syntax:
    
        aDekel(Mv,c2,z=0.)
    
    where
    
        Mv: halo mass [M_sun] (float or array)
        c2: concentration, c_-2 = R_vir / r_-2 (float or array of the 
            same size as Mv)
        z: redshift (float or array of the same size as Mv, default=0.)
            
    Return:
    
        Dekel+ alpha (float or array of the same size as Mv)
        stellar mass [M_sun] (float or array of the same size as Mv)
    """
    if z>6.: # a safety: in the regime where the stellar-halo mass
        # relations are not reliable, manually set the stellar mass
        Ms = 1e-5 * Mv 
    else: 
        Ms = Mstar(Mv,z)
    mu = gh.slope(Ms/Mv)
    s = np.maximum(0., np.random.normal(mu, 0.15))
    #s = np.random.random(len(mu))*2. # <<< test, draw s_0.01 randomly
        # between 0-2.
    r = np.sqrt(c2)
    return ( s + (2.*s-7.)*r/15. ) / (1.+(s-3.5)*r/15.), Ms

def cDekel(c2,alpha):
    """
    Compute the Dekel+ concentration, c, using the conventional 
    concentration, c_-2, and the Dekel+ innermost slope, alpha.
    
    Syntax:
    
        cDekel(c2,alpha)
        
    where
        
        c2: concentration, c_-2 = R_vir / r_-2 (float or array)
        alpha: Dekel+ innermost slope (float or array of the same size
            as c2)
    
    Return:
    
        Dekel+ concentration (float or array of the same size as c2)
    """
    return (2.-alpha)**2 / 2.25 * c2
    
def Dekel(Mv,z=0.):
    """
    Draw the Dekel+ structural parameters, c and alpha, as well as the 
    stellar mass, given the halo mass and redshift.
    
    Internally, it draws the conventional halo concentration, c_-2, which
    is used to compute alpha.
    
    Syntax:
    
        Dekel(Mv,z=0.)
        
    where
    
        Mv: halo mass [M_sun] (float or array)
        z: redshift (float or array of the same size as Mv, default=0.)
    
    Return:
        
        Dekel+ concentration (float or array of the same size as Mv), 
        Dekel+ alpha (float or array of the same size as Mv),
        stellar mass [M_sun] (float or array of the same size as Mv)
        c_-2 (float or array of the same size as Mv)
    """
    c2 = concentration(Mv,z)
    alpha, Ms = aDekel(Mv,c2,z)
    c = cDekel(c2,alpha)
    return c, alpha, Ms, c2
    
def Dekel_fromMAH(Mv,t,z):
    """
    Returns the Dekel+ structural parameters, c and alpha, given the 
    halo mass assembly history (MAH), using the Zhao+09 formula.
    
    Syntax:
        
        Dekel(Mv,t,z)
        
    where
    
        Mv: main-branch mass history until the time of interest [M_sun]
            (array)
        t: the time series of the main-branch mass history (array of the
            same length as Mv)
        z: the instantaneous redshift (float)
    
    Note that we need Mv and t in reverse chronological order, i.e., in 
    decreasing order, such that Mv[0] and t[0] are the instantaneous halo
    mass and instantaneous cosmic time, respectively.
    
    Return:
        
        Dekel+ concentration (float), 
        Dekel+ alpha (float), 
        stellar mass [M_sun] (float),
        c_-2 (float)
    """
    c2 = gh.c2_Zhao09(Mv,t)
    alpha, Ms = aDekel(Mv[0],c2,z)
    c = cDekel(c2,alpha)
    return c, alpha, Ms, c2
def c2_fromMAH(Mv,t,version='zhao'):
    """
    Returns the NFW concentration, c_{-2}, given the halo mass 
    assembly history (MAH), using the Zhao+09 formula.
    
    Syntax:
        
        c2_fromMAH(Mv,t,version)
        
    where
    
        Mv: main-branch mass history until the time of interest [M_sun]
            (array)
        t: the time series of the main-branch mass history (array of the
            same length as Mv)
        version: 'zhao' or 'vdb' for the different versions of the
                 fitting function parameters (string)
    
    Note that we need Mv and t in reverse chronological order, i.e., in 
    decreasing order, such that Mv[0] and t[0] are the instantaneous halo
    mass and instantaneous cosmic time, respectively.
    
    Return:
        
        c_-2 (float)
    """
    return gh.c2_Zhao09(Mv,t,version)
    
#---for initializing orbit

def orbit_from_Jiang2015(hp, sp, z, flat_2D=False):
    """
    Initialize the orbit of a satellite by sampling from V/V_{200c}
    and Vr/V distributions from Jiang+2015. Subhaloes are placed
    on initial orbital radii of r_{200c} of the host. This is an
    extension of the Jiang+15 model, as we use the host peak height,
    rather than host mass at z=0, in order to determine which
    distribution to sample from.
    
    Syntax:
    
        orbit_from_Jiang2015(hp, sp, z, flat_2D)
        
    where
    
        hp: host *NFW* potential (a halo density profile object, 
            as defined in profiles.py)
        sp: subhalo *NFW* potential (a halo density profile object, 
            as defined in profiles.py) 
        z: the redshift of accretion (float)
        flat_2D: set to True to initialize all haloes in z=0 plane, for
                 visualization purposes
            
    Return:
    
        phase-space coordinates in cylindrical frame 
        np.array([R,phi,z,VR,Vphi,Vz])

    Note:
        This assumes NFW profiles profile, since we're using the
        .otherMassDefinition() method that has only been implemented
        for NFW.
    """
    Mh200c, rh200c, ch200c = hp.otherMassDefinition(200.)
    Ms200c, rs200c, cs200c = sp.otherMassDefinition(200.)

    nu = co.nu(Mh200c, z, **cfg.cosmo)
    mass_ratio = Ms200c / Mh200c
    
    iM = np.searchsorted(cfg.jiang_nu_boundaries, nu)
    imM = np.searchsorted(cfg.jiang_ratio_boundaries, mass_ratio)

    rand_VV200c = np.random.uniform()
    rand_VrV = np.random.uniform()

    # NOTE: Can uncomment this block of code if you don't want to sample
    #       unbound orbits.
    #vbyvv200c_max = np.sqrt(2.*np.abs(hp.Phi(rh200c))) / hp.Vcirc(rh200c)
    #while True:
    #    rand_VV200c = np.random.uniform()
    #    V_by_V200c = splev(rand_VV200c, cfg.V_by_V200c_interps[iM][imM])
    #    if(V_by_V200c < vbyvv200c_max): # sample until we get a bound orbit
    #        break

    V_by_V200c = splev(rand_VV200c, cfg.V_by_V200c_interps[iM][imM])
    Vr_by_V = splev(rand_VrV, cfg.Vr_by_V_interps[iM][imM])
    gamma = np.pi - np.arccos(Vr_by_V)

    V0 = V_by_V200c * hp.Vcirc(rh200c)
    theta = np.arccos(2.*np.random.random()-1.) # i.e., isotropy
    zeta = 2.*np.pi*np.random.random() # i.e., uniform azimuthal 
        # angle, zeta, of velocity vector in theta-phi-r frame 
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    singamma = np.sin(gamma)
    cosgamma = np.cos(gamma)
    sinzeta = np.sin(zeta)
    coszeta = np.cos(zeta)
    if(flat_2D):
        return np.array([
            rh200c,
            np.random.random() * 2.*np.pi,  # uniformly random phi in (0,2pi)
            0.,
            V0 * cosgamma,
            np.random.choice([-1,1]) * V0 * singamma, # clock/counter-clockwise
            0.])
    else:
        return np.array([
            rh200c * sintheta,
            np.random.random() * 2.*np.pi,  # uniformly random phi in (0,2pi)
            rh200c * costheta,
            V0 * ( singamma * coszeta * costheta + cosgamma * sintheta ),
            V0 * singamma * sinzeta,
            V0 * ( cosgamma * costheta - singamma * coszeta * sintheta ),
            ])
    
def ZZLi2020(hp, Msub, z):
    """
    Compute the V/Vvir and infall angle of a satellite given the host
    and subhalo masses and the redshift of the merger based on the
    universal model of Zhao-Zhou Li, in prep.
    
    Syntax:
    
        ZZLi2020(hp, Msub, z)
        
    where
    
        hp: host potential (a halo density profile object, as defined 
            in profiles.py) 
        Msub: infalling subhalo mass (float)
        z: redshift of merger (float)
            
    Return:

        v_by_vvir: total velocity at infall, normalized by Vvir (float)
        gamma: angle of velocity vector at infall (radians, float)
    
    Note:
        Theta is defined to be zero when the subhalo is falling radially
        in. Hence, for consistency with our coordinate system, we return
        gamma = pi - theta, theta=0 corresponds to gamma=pi, radial infall.
    """
    Mhost = hp.Mh
    zeta = Msub / Mhost
    nu = co.nu(Mhost, z, **cfg.cosmo)
    A = 0.30*nu -3.33*zeta**0.43 + 0.56*nu*zeta**0.43
    B = -1.44 + 9.60*zeta**0.43

    # NOTE: Can uncomment this block of code if you don't want to sample
    #       unbound orbits.
    #vbyvv_max = np.sqrt(2.*np.abs(hp.Phi(hp.rh))) / hp.Vcirc(hp.rh)
    #while True:
    #    v_by_vvir = lognorm.rvs(s=0.22, scale=1.2)
    #    if(v_by_vvir < vbyvv_max): # sample until we get a bound orbit
    #        break

    v_by_vvir = lognorm.rvs(s=0.20, scale=1.20)
    eta = 0.89*np.exp(-np.log(v_by_vvir / 1.04)**2. / (2. * 0.20**2.)) + A*(v_by_vvir + 1) + B

    if(eta <= 0): 
        one_minus_cos2t = np.random.uniform()
    else:
        cum = np.random.uniform(0.0, 0.9999) # cut right below 1, avoids 1-cos2t>1
        one_minus_cos2t = (-1. / eta) * np.log(1. - cum*(1. - np.exp(-eta)))
    theta = np.arccos(np.sqrt(1. - one_minus_cos2t))
    # TODO: Can change above to repeat if it yields a NaN theta, but this is quite rare
    assert ~np.isnan(theta), "NaN theta, 1-cos^2t=%.1f, z=%.2f, Mhost=%.2e, Msub=%.2e" %\
            (one_minus_cos2t, z, Mhost, Msub)
    gamma = np.pi - theta
    return v_by_vvir, gamma
def orbit_from_Li2020(hp, vel_ratio, gamma, flat_2D=False):
    """
    Initialize the orbit of a satellite, given total velocity V/Vvir 
    and infall angle.  
    
    Syntax:
    
        orbit(hp, vel_ratio, gamma, flat_2D)
        
    where
    
        hp: host potential (a halo density profile object, as defined 
            in profiles.py) 
        vel_ratio: the total velocity at infall in units of Vvir
        gamma: the angle between velocity and position vectors of subhalo
        flat_2D: set to True to initialize all haloes in z=0 plane, for
                 visualization purposes
            
    Return:
    
        phase-space coordinates in cylindrical frame 
        np.array([R,phi,z,VR,Vphi,Vz])

    Note:
        This assumes that the BN98 virial mass definition is used
        for the haloes, since the host rh quantity is used as the radius
        where the circular velocity is computed.
    """
    r0 = hp.rh
    V0 = vel_ratio * hp.Vcirc(r0)
    theta = np.arccos(2.*np.random.random()-1.) # i.e., isotropy
    zeta = 2.*np.pi*np.random.random() # i.e., uniform azimuthal 
        # angle, zeta, of velocity vector in theta-phi-r frame 
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    singamma = np.sin(gamma)
    cosgamma = np.cos(gamma)
    sinzeta = np.sin(zeta)
    coszeta = np.cos(zeta)
    if(flat_2D):
        return np.array([
            r0,
            np.random.random() * 2.*np.pi,  # uniformly random phi in (0,2pi)
            0.,
            V0 * cosgamma,
            np.random.choice([-1,1]) * V0 * singamma, # clock/counter-clockwise
            0.])
    else:
        return np.array([
            r0 * sintheta,
            np.random.random() * 2.*np.pi,  # uniformly random phi in (0,2pi)
            r0 * costheta,
            V0 * ( singamma * coszeta * costheta + cosgamma * sintheta ),
            V0 * singamma * sinzeta,
            V0 * ( cosgamma * costheta - singamma * coszeta * sintheta ),
            ])

def orbit(hp,xc=1.0,eps=0.5, flat_2D=False):
    """
    Initialize the orbit of a satellite, given orbit energy proxy (xc) 
    and circularity (eps).  
    
    Syntax:
    
        orbit(hp,xc=1.,eps=0.5,)
        
    where
    
        hp: host potential (a halo density profile object, as defined 
            in profiles.py) 
        xc: the orbital energy parameter, defined such that if the 
            energy of the orbit is E, x_c(E) is the radius of a circular 
            orbit in units of the host halo's virial radius (default=1.)
        eps: the orbital circularity parameter (default=0.5)
        flat_2D: set to True to initialize all haloes in z=0 plane, for
                 visualization purposes
            
    Return:
    
        phase-space coordinates in cylindrical frame 
        np.array([R,phi,z,VR,Vphi,Vz])
    """
    r0 = hp.rh
    rc = xc * hp.rh
    theta = np.arccos(2.*np.random.random()-1.) # i.e., isotropy
    zeta = 2.*np.pi*np.random.random() # i.e., uniform azimuthal 
        # angle, zeta, of velocity vector in theta-phi-r frame 
    Vc = hp.Vcirc(rc,)
    Phic = hp.Phi(rc,)
    Phi0 = hp.Phi(r0,)
    V0 = np.sqrt(Vc**2 + 2.*(Phic-Phi0)) 
    S = eps * rc/r0 * Vc/V0
    gamma = np.pi-np.arcsin(S) # angle between r and v vectors. Note that
        # we use pi - np.arcsin(S) instead of just np.arcsin(S), because 
        # the velocity needs to point inward the virial sphere.
    if S>1.: # a safety, may not be useful
        sys.exit('Invalid orbit! sin(gamma)=%.4f,xc=%4.2f,eps=%4.2f'\
            %(S,xc,eps))
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    singamma = np.sin(gamma)
    cosgamma = np.cos(gamma)
    sinzeta = np.sin(zeta)
    coszeta = np.cos(zeta)
    if(flat_2D):
        return np.array([
            r0,
            np.random.random() * 2.*np.pi,  # uniformly random phi in (0,2pi)
            0.,
            V0 * cosgamma,
            np.random.choice([-1,1]) * V0 * singamma, # clock/counter-clockwise
            0.])
    else:
        return np.array([
            r0 * sintheta,
            np.random.random() * 2.*np.pi,  # uniformly random phi in (0,2pi)
            r0 * costheta,
            V0 * ( singamma * coszeta * costheta + cosgamma * sintheta ),
            V0 * singamma * sinzeta,
            V0 * ( cosgamma * costheta - singamma * coszeta * sintheta ),
            ])
