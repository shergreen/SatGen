##################### cosmology-related functions #######################

# Arthur Fangzhou Jiang 2019 Hebrew University

#########################################################################

import config as cfg

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
import cosmolopy.distance as cdis
import cosmolopy.density as cden
import cosmolopy.constants as cc
import cosmolopy.perturbation as cper

#########################################################################

#---basics 

def rhoc(z,h=0.7,Om=0.3,OL=0.7):
    """
    Critical density [Msun kpc^-3] at redshift z.
    
    Syntax: 
    
        rhoc(z,h=0.7,Om=0.3,OL=0.7)
    
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return cfg.rhoc0 * h**2 * (Om*(1.+z)**3 + OL)
    
def rhom(z,h=0.7,Om=0.3,OL=0.7):
    """
    Mean density [Msun kpc^-3] at redshift z.
    
    Syntax: 
    
        rhom(z,h=0.7,Om=0.3,OL=0.7)
    
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return Omega(z,Om,OL) * rhoc(z,h,Om,OL)
    
def DeltaBN(z,Om=0.3,OL=0.7):
    """
    Virial overdensity of Bryan & Norman (1998).
    
    Syntax:
    
        DeltaBN(z, Om=0.3,OL=0.7)
        
    where
        
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    x = Omega(z,Om,OL) - 1.
    return 18.*np.pi**2 + 82.*x - 39.*x**2

def Omega(z,Om=0.3,OL=0.7):
    """
    Matter density in units of the critical density, at redshift z.
    
    Syntax: 
    
        Omega(z, Om=0.3,OL=0.7)
        
    where
        
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    fac = Om * (1.+z)**3
    return fac / (fac + OL)

def tdyn(z,h=0.7,Om=0.3,OL=0.7):
    """
    Halo dynamical time [Gyr] defined as
    
        R_vir/V_vir = sqrt(3 / 4 pi G Delta rho_crit) 
                    = sqrt(2 / Delta) * 1/H(z)
        
    Syntax: 
    
        tdyn(z,h=0.7,Om=0.3,OL=0.7)
    
    where 
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    
    Note that at high-z, this is rougly 0.1 times Hubble time 1/H(z).    
    """
    return np.sqrt(2./DeltaBN(z,Om,OL)) / H(z,h,Om,OL)

def Ndyn(z1,z2,h=0.7,Om=0.3,OL=0.7):
    """
    Number of halo dynamical times elapsed between redshift z1 and z2 
    (z1>z2).
    
    Syntax: 
    
        Ndyn(z1,z2,h=0.7,Om=0.3,OL=0.7)
        
    where
    
        z1: a higher redshift (float)
        z2: a lower redshift
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return quad(dNdz, z1,z2, args=(h,Om,OL,),
        epsabs=1.e-7, epsrel=1.e-6,limit=10000)[0]
def dNdz(z,h,Om,OL):
    r"""
    Auxiliary function for the function Ndyn -- the integrand, dN/dz(z),
    for computing 
    
        N_dyn = int_z1^z2 dN/dz(z) dz
              = int_z1^z2 dt/dz(z) * 1/t_dyn(z) dz
    
    Syntax:
        
        dNdz(z,h,Om,OL)
    
    where
        
        z: redshift (float)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return dtdz(z,h,Om,OL) / tdyn(z,h,Om,OL)
def dtdz(z,h,Om,OL):
    """
    complementary function for computing N_dyn, it returns
        dt / dz
    i.e., cosmic time increment per redshift decrement 
    """
    z1 = z*(1.-cfg.eps)
    z2 = z*(1.+cfg.eps)
    t1 = t(z1,h,Om,OL) # t1>t2 because z1<z2
    t2 = t(z2,h,Om,OL)
    return (t1-t2) / (z1-z2)

def H(z,h=0.7,Om=0.3,OL=0.7):
    """
    Hubble constant [Gyr^-1] at redshift z.
    
    Syntax:
    
        H(z,h=0.7,Om=0.3,OL=0.7)
        
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return (h/9.778) * np.sqrt( Om*(1.+z)**3 + OL )

def E(z,Om=0.3,OL=0.7):
    """
    Hubble constant at redshift z in units of the Hubble constant at z=0.
        
        E(z):=H(z)/H0

    Syntax:
    
        E(z,Om=0.3,OL=0.7)
        
    where
    
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return np.sqrt( Om*(1.+z)**3 + OL )
    
def t(z,h=0.7,Om=0.3,OL=0.7):
    r"""
    Cosmic time [Gyr] (time since Big Bang).
    
    Syntax: 
    
        t(z,h=0.7,Om=0.3,OL=0.7)
    
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    fac = OL / (1.+z)**3
    return (9.778/h) * 2./(3.*np.sqrt(OL)) * \
        np.log((np.sqrt(fac)+np.sqrt(fac+Om)) / np.sqrt(Om))

def tlkbk(z,h=0.7,Om=0.3,OL=0.7):
    """
    Lookback time [Gyr] at redshift z.
    
    Syntax: 
    
        tlkbk(z,h=0.7,Om=0.3,OL=0.7)
    
    where
        
        z: redshift (float or array)
        h: dimensionless Hubble constant at z=0, defined in
            H_0 = 100h km s^-1 Mpc^-1 
                = h/10 km s^-1 kpc^-1 
                = h/9.778 Gyr^-1 
            (default=0.7)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
        OL: dark-energy density in units of the critical density, at z=0
            (default=0.7) 
    """
    return t(0.,h,Om,OL) - t(z,h,Om,OL) 

#------------------------- for EPS formalism ----------------------------
# - critical overdensity for collapse, 
# - transfer function, 
# - linear power spectrum, 
# - mass variance,
# - peak height,
# - Parkinson+08 algorithm
# - EPS conditional mass function & progenitor mass function
#   They all depend on the CosmoloPy library, and thus are grouped here. 

# critical overdensity for collapse
def deltac(z,Om=0.3):
    """
    Critical linearized overdensity for spherical collapse.
    
    Syntax:
    
        delta_coll(z,Om=0.3)
        
    where
        
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
    """
    return 1.686 / D(z,Om)
def D(z,Om=0.3):
    """
    Linear growth rate D(z).
    
    Syntax:
    
        D(z,Om=0.3)
    
    where
        
        z: redshift (float or array)
        Om: matter density in units of the critical density, at z=0
            (default=0.3) 
    """
    return cper.fgrowth(z,Om) 

# transfer function    
def T(k, **cosmo):
    """
    Transfer function of Eisenstein & Hu (1999 ApJ 511 5), with optional
    baryonic effects of Eisenstein & Hu (1997 ApJ 496 605), as 
    implemented in the CosmoloPy library.
    
    Syntax:
    
        T(k, **cosmo)
        
    where
    
        k: wave number [h Mpc^-1] (float or array)
        cosmo: cosmological parameters (dictionary defined in config.py)
    
    Note that if cosmo['m_WDM'] exists, multiply a correction factor, 
    following Bode+01 as cited by Lovell+14, to account for WDM effect.
    """ 
    h = cosmo['h']
    k = h*k # CosmoloPy takes k in [Mpc^-1]
    Ttmp = cper.transfer_function_EH(k, **cosmo)[0]
    if 'm_WDM' in cosmo:
        a = 0.05 * cosmo['m_WDM']**(-1.15) * \
            (cosmo['omega_M_0']/0.4)**0.15 * \
            (h/0.65)**1.3
        Ttmp = Ttmp * (1. + (a*k)**2 )**(-5)
    return Ttmp

# power spectrum
def P(k,z=0.,**cosmo):
    """
    Power spectrum. 
    
    Syntax:
    
        P(k,z=0.,**cosmo)
    
    where 
    
        k: wave number [h Mpc^-1] (float or array)
        z: redshift (default=0.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    Om = cosmo['omega_M_0']
    ns = cosmo['n']
    if 'k0' not in cosmo: # i.e., not normalized yet
        return (T(k,**cosmo)*D(z,Om))**2. * (k/k0(**cosmo))**ns
    else:                 # i.e., already normalized to sigma_8
        return (T(k,**cosmo)*D(z,Om))**2. * (k/cosmo['k0'])**ns
def k0(**cosmo):
    """
    Normalization (k_0) of the primordial power spectrum 
    
        P_primordial(k) = (k/k_0)^n
    
    such that 
    
        sigma(R=8Mpc/h,z=0) = simga_8.
        
    Syntax:
    
        k0(**cosmo)
        
    where
    
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    cosmo['k0'] = 1./3000. # give a temporary, arbitrary normalization 
    k0tmp = cosmo['k0']
    s8 = cosmo['sigma_8']
    ns = cosmo['n']
    s8tmp = sigmaR(8.,**cosmo)
    return k0tmp * (s8tmp / s8 )**(2./ns)
def sigmaR(R,**cosmo):
    """
    Variance of density field smoothed over a spatial scale, linearly 
    extrapolated to z=0.
    
    That is, the integral of 
    
        k^3 / (2 pi^2) * P(k,z=0) * W(k,R)^2 dln(k)
        
    from ln(k) = -inf to +inf, where W(k,R) is the F.T of a window 
    function of size R.
    
    Syntax: 
    
        sigmaR(R,**cosmo)
    
    where
    
        R: the comoving spatial scale of interest [Mpc/h] (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
        
    Return:
    
        the sqrt of the variance, i.e., sigma(R) (float) 
    """
    lnkc = np.log(1./R) # discontinuity of the integrand at ln(k_c)
    # divide the integral range at the discontinuity, ln(k_c), and 
    # integrate separately for the two parts
    S1, S1err = quad(dSdlnk, -50., lnkc, args=(R,cosmo),
                     epsabs=1e-7, epsrel=1e-6,limit=10000)
    S2, S2err = quad(dSdlnk, lnkc, 50., args=(R,cosmo),
                     epsabs=1e-7, epsrel=1e-6,limit=10000)
    S = S1+S2
    return np.sqrt(S)
def dSdlnk(lnk,R,cosmo):
    """
    Auxiliary function -- the integrand for "sigmaR".
    """
    k = np.exp(lnk)
    return DeltaSqr(k,z=0.,**cosmo) * W(k,R)**2
def DeltaSqr(k,z=0.,**cosmo):
    """
    Dimensionless power spectrum,
    
        Delta(k)^2 := k^3 P(k) / (2 pi^2), 
        
    which represents the contribution per log wavenumber of the power 
    spectrum to the variance.
    
    Syntax:
    
        DeltaSqr(k,z=0.,**cosmo)
    
    where 
    
        k: wave number [h Mpc^-1] (float or array)
        z: redshift (default=0.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return k**3 / cfg.TwoPisqr * P(k,z,**cosmo)
def W(k,R):
    """
    F.T. of a spherical tophat window function of a given spatial scale. 
    
    Syntax:
        
        W(k,R)
    
    where
        
        k: wave number [h Mpc^-1] (float or array)
        R: the comoving spatial scale of interest [Mpc/h] (float)
    """
    x = k*R
    j1 = (np.sin(x) - x*np.cos(x)) / x**2.
    return 3.*j1/x

# mass variance
def sigma(M,z=0.,**cosmo):
    """
    Variance of linearized density field, smoothed over a mass scale.
    
    Syntax:
    
        sigmaM(M,z=0.,**cosmo)
        
    where
    
        M: the mass scale of interest [M_sun] (float)
        z: redshift (default=0.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    
    Note that this is a wrapper of the function "sigmaM". "sigmaM" only 
    takes a single mass as the input. This function can also take an 
    array of masses as the input, and return an array of sigma(M).
    
    Return:
    
        the sqrt of the variance, i.e., sigma(M,z) (float or array) 
    """
    Om = cosmo['omega_M_0']
    if np.isscalar(M):
        return sigmaM(M,**cosmo) * D(z,Om)
    else:
        return sigmaM_vec(M,**cosmo) * D(z,Om)
def sigmaM(M,**cosmo):
    """
    Variance of density field smoothed over a mass scale, linearly 
    extrapolated to z=0.
    
    Syntax:
    
        sigmaM(M,**cosmo)
        
    where
    
        M: the mass scale of interest [M_sun] (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
        
    Return:
    
        the sqrt of the variance, i.e., sigma(M) (float) 
    """
    h = cosmo['h']
    Om = cosmo['omega_M_0']
    OL = cosmo['omega_lambda_0']
    if cosmo['MassVarianceChoice']==0:
        rho=rhom(0.,h,Om,OL) # [Msun kpc^-3]
        R = ( M / rho / cfg.FourPiOverThree )**(1./3.) # [kpc]
        R = R / 1000. * h # sigmaR takes R in [Mpc/h]
        return sigmaR(R,**cosmo)
    else:
        return cfg.sigmalgM_interp(np.log10(M))
sigmaM_vec = np.vectorize(sigmaM, doc="Vectorized 'sigmaM'")

# peak height
def nu(M,z=0,**cosmo):
    """
    Peak height, 
    
        delta_c / sigma(M,z)
    
    where delta_c = 1.686 is the critical overdensity for spherical 
    tophat collapse, and sigma(M,z) is the RMS density fluctuation in 
    spherical tophats of mass M at redshift z.
    
    Syntax:
    
        nu(M,z=0,**cosmo)
        
    where:
        
        M: the mass scale of interest [M_sun] (float or array)
        z: redshift (default=0.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return 1.686 / sigma(M,z,**cosmo)

# Parkinson+08 algorithm  

def dlnSdlnM(M,**cosmo):
    """
    The derivative of mass variance:
    
        dln(S)/dln(M) 
    
    where S=sigma(M,z=0)^2 is the mass variance linearly extrapolated to
    z=0.
    
    Syntax:
    
        dlnSdlnM(M,**cosmo)
    
    where
    
        M: the mass scale of interest [M_sun] (float or array)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return 2.* dlnsigmadlnM(M,**cosmo)
def dlnsigmadlnM(M,**cosmo):
    """
    The derivative of the RMS density fluctuation in spherical tophats of 
    mass M:
    
        dln[sigma(M,z=0))]/dln(M) 
    
    Syntax:
    
        dlnSdlnM(M,**cosmo)
    
    where
    
        M: the mass scale of interest [M_sun] (float or array)
        cosmo: cosmological parameters (dictionary defined in config.py)
        
    Note that the Parkinson+08 alpha(M) factor is the absolute value,
    i.e., the negative, of this function.
    """
    M1 = (1.+cfg.eps)*M
    M2 = (1.-cfg.eps)*M
    sigma1 = sigma(M1,0.,**cosmo)
    sigma2 = sigma(M2,0.,**cosmo)
    return (np.log(sigma1) - np.log(sigma2))/(np.log(M1) - np.log(M2))

def UpdateGlobalVariables(**cosmo):
    """
    Update a few intermediate global variables that are repeatedly used 
    by the functions for the Parkinson+08 algorithm.
    
    Syntax:
    
        UpdateGlobalVariables(**cosmo)
        
    where 
    
        cosmo: cosmological parameters (dictionary defined in config.py)    
    """
    cfg.W0 = deltac(cfg.z0,cosmo['omega_M_0'])
    if cfg.M0>cfg.Mres:
        cfg.qres = min(cfg.Mres/cfg.M0,0.499) # 0.499 is a safety
    else:
        cfg.qres = min(cfg.Mmin/cfg.M0,0.499)
    cfg.sigmares = sigma(cfg.qres*cfg.M0,0.,**cosmo)
    cfg.sigma0 = sigma(cfg.M0,0.,**cosmo)
    cfg.sigmah = sigma(0.5*cfg.M0,0.,**cosmo)
    cfg.S0 = cfg.sigma0**2
    cfg.Sh = cfg.sigmah**2
    Sres = cfg.sigmares**2
    cfg.alphah = -dlnsigmadlnM(0.5*cfg.M0,**cosmo)
    cfg.ures = cfg.sigma0/np.sqrt(Sres-cfg.S0)
    Vres = Sres / (Sres - cfg.S0)**1.5
    Vh = cfg.Sh / (cfg.Sh - cfg.S0)**1.5
    cfg.beta = np.log(Vres/Vh) / np.log(2.*cfg.qres)
    cfg.B = 2.0**cfg.beta * Vh
    cfg.mu = cfg.alphah if cfg.gamma1>=0. else \
        - np.log(cfg.sigmares/cfg.sigmah) / np.log(2.*cfg.qres)
    cfg.eta = cfg.beta - 1. - cfg.gamma1*cfg.mu
    cfg.NupperOverdW = NupperOverdW()
    cfg.dW = dW()
   
def R(q,**cosmo): 
    """
    The factor 
    
        R(q),
        
    as in Parkinson+08 Eq.(A3).
    
    Syntax:
        
        R(q,**cosmo)
    
    where
        
        q: M_1 / M_0, where M_1 is the mass of a progenitor of M_0
            (float or array)
        cosmo: cosmological parameters (dictionary defined in config.py)
        
    Note that this function uses global variables.
    """
    M1 = q*cfg.M0
    S1 = sigma(M1,0.,**cosmo)**2
    V = S1 / (S1 - cfg.S0)**1.5
    fac1 = -dlnsigmadlnM(M1,**cosmo) / cfg.alphah
    fac2 = V / (cfg.B * q**cfg.beta)
    fac3=((2.*q)**cfg.mu *sigma(M1,0.,**cosmo)/cfg.sigmah)**cfg.gamma1
    Rtmp = fac1 * fac2 * fac3
    #if Rtmp>1.0: # <<< a safety check, may remove if turned out useless
    #    print("Warning: R(q=%g)=%g>1, fac1=%g,fac2=%g,fac3=%g"%\
    #        (q,Rtmp,fac1,fac2,fac3))
    return Rtmp
    
def dW():
    """
    Timestep for the Parkinson+08 method.
    
    Syntax:
        
        dW()
        
    Note that this function uses global variables.
    """
    dW1 = 0.1 * cfg.Root2 * np.sqrt(cfg.Sh-cfg.S0)
    dW2 = 0.1 / cfg.NupperOverdW
    return min(dW1,dW2)
def NupperOverdW():
    """
    The integral used to determine the timestep, dW, i.e.,
    
        N_upper/dW, 
        
    where N_upper is given by Parkinson+08 Eqs.(A3),(A5).
    
    Syntax:
        
        NupperOverdW()
        
    Note that this function uses global variables.
    """
    A = cfg.Root2OverPi * cfg.B * cfg.alphah * cfg.G0 \
        / 2.**(cfg.mu*cfg.gamma1) * (cfg.W0/cfg.sigma0)**cfg.gamma2 \
        * (cfg.sigmah/cfg.sigma0)**cfg.gamma1 # this is the Parkinson+08
        # S(q) in Eq.(A2) apart from the factor q^(eta-1).  
    if cfg.qres>=(0.5-cfg.eps):
        I = cfg.eps
    else:
        if np.abs(cfg.eta)>cfg.eps:
            I = (0.5**cfg.eta - cfg.qres**cfg.eta)/cfg.eta
        else:
            I = - np.log(2.*cfg.qres)
    return A * I

def J(ures):
    r"""
    J(u_res), as given by Eq.(A7) of Parkinson+08.
    
    Syntax:
    
        J(ures)
        
    where:
    
        ures: sigma(M_0) / [S(M1)-S(M0)]^1/2 (float)
    """
    return quad(dJdu,0.,ures,epsabs=1e-7,epsrel=1e-6,limit=50)[0]
J_vec = np.vectorize(J, doc="Vectorized 'J(u_res)' function")
def dJdu(u):
    """
    Integrand of J.
    
    Note that this function uses the global variable, cfg.gamma1.
    """
    return (1.+1./u**2)**(cfg.gamma1/2.)
    
def F():
    """
    The smooth accretion fraction, M_smooth / M_0, during a timestep dW,
    as in Parkinson+08 eq.(A6).
    
    Syntax:
    
        F()
    
    Note that this function uses global variables.
    """
    return min(0.5, cfg.Root2OverPi * cfg.Jures_interp(cfg.ures) * \
           cfg.G0/cfg.sigma0 * (cfg.W0/cfg.sigma0)**cfg.gamma2 * cfg.dW)
           
def DrawProgenitors(**cosmo):
    """
    Draw progenitor masses using the Parkinson+08 method.
    
    Syntax:
         
         DrawProgenitors(**cosmo)
         
    where
    
        cosmo: cosmological parameters (dictionary defined in config.py)
         
    Return 
        
        mass of main progenitor (float),
        mass of secondary progenitor (float, =0. if only one progenitor),
        number of progenitors (int, either 1 or 2)
    """
    r1 = np.random.random()
    Nupper = cfg.NupperOverdW * cfg.dW
    Np = 0 # initialize
    if r1 > Nupper:
        M1 = cfg.M0 * (1.-F())
        M2 = 0.
    else:
        r2 = np.random.random()
        q = (cfg.qres**cfg.eta + \
            r2*(2.**(-cfg.eta) - cfg.qres**cfg.eta))**(1./cfg.eta)
        r3 = np.random.random()
        if (r3<R(q,**cosmo)):
            Mtmp1 = cfg.M0 * (1.-F()-q)
            Mtmp2 = cfg.M0 * q
            M1 = max(Mtmp1,Mtmp2)
            M2 = min(Mtmp1,Mtmp2)
        else:
            M1 = cfg.M0 * (1.-F()) 
            M2 = 0.
    if M1>cfg.Mres: Np += 1
    if M2>cfg.Mres: Np += 1
    return M1,M2,Np

# EPS conditional mass function & progenitor mass function

def Masterisk(z=0.,height=1.,**cosmo):
    """
    The Press-Schechter mass, or, more generally, the mass corresponding 
    to a density peak of a given height at a given redshift [M_sun].
    
    Syntax:
    
        Masterisk(z=0.,nu=1.,**cosmo)
    
    where
        
        z: redshift (default=0.)
        height: peak height (default=1.)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return brentq(FindMasterisk, 1e1, 1e17, args=(z,height,cosmo), 
        xtol=1e-5, rtol=1e-3, maxiter=100)
def FindMasterisk(M,z,height,cosmo):
    """
    Auxiliary function for the function "Masterisk".
    """
    return nu(M,z,**cosmo) - height
    
def dNdlnM1(M1,z1,M0,z0,**cosmo):
    """
    The EPS progenitor mass function (PMF), 
    
        dN/dln(M_1) = M_0/M_1 dP/dln(M_1),
        
    i.e., the mean number of progenitors with mass in the logarithmic 
    mass bin [ln(M_1), ln(M_1)+dln(M_1)].
    
    Strictly speaking, the PMF is dN/dM_1, related to dN/dln(M_1) by 
    
        dN/dM_1 = 1/M_1 * dN/dln(M_1) := n_EPS(M_1,z_1|M_0,z_0)
    
    where n_EPS(M_1,z_1|M_0,z_0)dM_1 is the mean number of progenitors of 
    mass [M_1,M_1+dM_1] at redshift z_1, which belong to descendent 
    halos of mass M_0 at redshift z_0, as defined in, e.g., 
    Jiang & van den Bosch (2014) Eq.(3).
    
    Syntax:
    
        dNdlnM1(M1,z1,M0,z0,**cosmo)
        
    where
    
        M1: progenitor mass [M_sun] (float or array)
        z1: progenitor redshift (float)
        M0: descendent mass [M_sun] (float)
        z0: descendent redshift (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    return M0/M1*dPdlnM1(M1,z1,M0,z0,**cosmo)
def dPdlnM1(M1,z1,M0,z0,**cosmo):
    """
    The EPS conditional mass function (CMF), 
    
        dP/dln(M_1), 
        
    or equivalently,
    
        dP/dln(M_1/M_0).
    
    Strictly speaking, the CMF is dP/dM_1, related to dP/dln(M_1) by
    
        dP/dM_1 = 1/M_1 * dP/dln(M_1) := P(M_1,z_1|M_0,z_0),
    
    where P(M_1,z_1|M_0,z_0)dM_1 is the mass fraction of halos of mass 
    M_0 at redshift z_0 that was contained in progenitors of mass 
    [M_1,M_1+dM_1] at z_1>z_0, as defined in e.g., 
    Jiang & van den Bosch (2014) Eq.(1).
    
    Syntax:
    
        dNdlnM1(M1,z1,M0,z0,**cosmo)
        
    where
    
        M1: progenitor mass [M_sun] (float or array)
        z1: progenitor redshift (float)
        M0: descendent mass [M_sun] (float)
        z0: descendent redshift (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    Om = cosmo['omega_M_0']
    S1 = sigma(M1,0.,**cosmo)**2.
    S0 = sigma(M0,0.,**cosmo)**2.
    W1 = deltac(z1,Om)
    W0 = deltac(z0,Om)
    return fEPS(S1,W1,S0,W0) * S1 * (-dlnSdlnM(M1,**cosmo))
def fEPS(S1,W1,S0,W0):
    """
    The conditional probability density 
    
         f_EPS(S_1,W_1|S_0,W_0)
         
    as in f_EPS(S_1,W_1|S_0,W_0)dS_1, which is the probability for a 
    random walk passing through (S_0,W_0) to excute a first-upcrossing of 
    a higher overdensity barrier, W=W_1, at [S1,S1+dS1].
    
    Syntax:
    
        fEPS(S1,W1,S0,W0)
        
    where
    
        S1: variance of density fluctuations on the mass scale of the 
            progenitor mass, M_1, linearly extrapolated to z=0 
            (float or array)
        W1: critical overdensity for collapse at redshift z_1 (float)
        S0: variance of density fluctuations on the mass scale of the 
            descendent mass, M_0, linearly extrapolated to z=0 (float)
        W0: critical overdensity for collapse at redshift z_0 (float)
        
    Note that what we implement here is NOT the original 
    spherical-collapse conditional probability density function, but an 
    empirical fit to the Millenium simulation by Cole+08 Eq.(7). The 
    reason is that this function and the functions calling it are used as 
    benchmarks for testing if Monte-Carlo merger trees are accurate 
    compared to merger trees from simulations. While we don't think the 
    Millenium result is the ultimate representation of trees in 
    simulations (as the halo finding and linking procedure of the 
    Millenium simulations are not necessarily optimal), it is a commonly 
    used benchmark (e.g., by Parkinson+08 and Jiang & van den Bosch 14). 
    If a newer fit based on better simulations becomes available, we 
    shall replace the Cole+08 fit with the newer result. 
    """
    DeltaS = S1-S0
    v10 = (W1-W0)/np.sqrt(DeltaS)
    return 0.2 *v10**0.75 /DeltaS *np.exp(-0.1 *v10**3) # empirical 
    #return cfg.Root1Over2Pi*v10/DeltaS*np.exp(-0.5*v10**2) # SC EPS
    
def NGTM1(M1,z1,M0,z0,**cosmo):
    """
    The cumulative EPS progenitor mass function, 
    
        N(>M_1,z_1|M_0,z_0).
    
    Syntax:
    
        NGTM1(M1,z1,M0,z0,**cosmo)
    
    where:
    
        M1: progenitor mass [M_sun] (float or array)
        z1: progenitor redshift (float)
        M0: target halo mass [M_sun] (float)
        z0: target halo redshift (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    a = np.log(M1)
    b = np.log(M0)
    return quad(dNGTM1dlnM1, a, b, args=(z1,M0,z0,cosmo),
        epsabs=1e-4, epsrel=1e-3,limit=100)[0]
def dNGTM1dlnM1(lnM1,z1,M0,z0,cosmo):
    """
    The integrand for "NGTM1".
    """
    M1 = np.exp(lnM1)
    return dNdlnM1(M1,z1,M0,z0,**cosmo)
    
def MGTM1(M1,z1,M0,z0,**cosmo):
    """
    The cumulative mass-weighted EPS progenitor mass function, 
    
        M(>M_1,z_1|M_0,z_0).
    
    Syntax:
    
        MGTM1(M1,z1,M0,z0,**cosmo)
    
    where:
    
        M1: progenitor mass [M_sun] (float or array)
        z1: progenitor redshift (float)
        M0: target halo mass [M_sun] (float)
        z0: target halo redshift (float)
        cosmo: cosmological parameters (dictionary defined in config.py)
    """
    a = np.log(M1)
    b = np.log(M0)
    return quad(dMGTM1dlnM1, a, b, args=(z1,M0,z0,cosmo),
        epsabs=1e-4, epsrel=1e-3,limit=100)[0]
def dMGTM1dlnM1(lnM1,z1,M0,z0,cosmo):
    """
    The integrand for "MGTM1".
    """
    M1 = np.exp(lnM1)
    return M1*dNdlnM1(M1,z1,M0,z0,**cosmo)
 
# unevolved subhalo mass functions
   
def dNdlnmaM0_all(x,gamma,alpha,beta,zeta):
    """
    Jiang & van den Bosch (2014) fitting function for the total
    unevolved subhalo mass function, 
    
        dN/dln(x) = gamma x^alpha exp(-beta x^zeta)
    
    with x: = m_acc/M_0 the mass at infall divided by the host mass.
    
    Syntax:
    
        dNdlnmaM0_all(x,gamma,alpha,beta,zeta)
        
    where
    
        x: m_acc / M_0
        gamma: normalization
        alpha: slope
        beta: parameter for the location of decay
        zeta: parameter for the steepness of decay
    """
    return gamma* x**alpha * np.exp(-beta*x**zeta)

def dNdlnmaM0_1st(x,gamma1,gamma2,alpha1,alpha2,beta,zeta):
    """
    Jiang & van den Bosch (2014) fitting function for the 1st-order 
    unevolved subhalo mass function, 
    
        dN/dln(x) = [gamma1 x^alpha1 + gamma2 x^alpha2] exp(-beta x^zeta)
    
    with x: = m_acc/M_0 the mass at infall divided by the host mass.
    
    Syntax:
    
        dNdlnmaM0_1st(x,gamma1,gamma2,alpha1,alpha2,beta,zeta)
        
    where
    
        x: m_acc / M_0
        gamma1: normalization for the first power-law component
        alpha1: slope of the first power law
        gamma2: normalization for the second power-law component
        alpha2: slope of the second power law
        beta: parameter for the location of decay
        zeta: parameter for the steepness of decay
    """
    return (gamma1*x**alpha1+gamma2*x**alpha2)*np.exp(-beta*x**zeta)

def fsub_pred(M0, z0, level=1, **cosmo):
    """
    Predicted value for f_{sub} based on N_dyn, the dynamical age of
    a halo. Uses the method described in Section 4.2 of Jiang &
    van den Bosch (2016), "Paper 1".
        
    Syntax:
    
        fsub_pred(M0, z0,level,**cosmo)
        
    where
    
        M0: halo mass at redshift of observation (float)
        z0: halo redshift of observation (float)
        level: 1 or 2, i.e., the fraction of mass bound into
               level 1+ subhaloes or level 2+ subhaloes
        cosmo: dictionary of cosmological parameters

    Note:

        The Ndyn function called computes the number of dynamical
        times between two redshifts, but the dynamical time differs
        from that defined in Jiang & van den Bosch (2016) by a factor
        of pi/2.
    """
    Om = cosmo['omega_M_0']
    OL = cosmo['omega_lambda_0']
    h = cosmo['h']
    f = 0.5
    alpha_f = 0.815 * np.exp(-2. * f**3.) / f**0.707
    omegat_f = np.sqrt(2.*np.log(alpha_f + 1.))
    rhs = deltac(z0,Om) + omegat_f * np.sqrt(
          sigma(f*M0,**cosmo)**2. - sigma(M0,**cosmo)**2.)
    eqn = lambda zf: deltac(zf,Om) - rhs
    zform = brentq(eqn, 0., 1000., 
            xtol=1e-5, rtol=1e-3, maxiter=100)
    Nt = Ndyn(zform,z0,h,Om,OL) / (np.pi / 2.)

    if(level == 1):
        return 0.325/Nt**0.6 - 0.075
    elif(level == 2):
        return 0.0461/Nt**1.3 - 0.0035
    else:
        sys.exit("Invalid subhalo level chosen for fsub!")