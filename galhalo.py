################## Functions for galaxy-halo connection ####################

# Arthur Fangzhou Jiang 2019, HUJI
# Sheridan Beckwith Green 2020, Yale

#########################################################################

import numpy as np

import config as cfg
import aux

#########################################################################

#---galaxy-size-halo-structure relation   

def Reff(Rv,c2):
    """
    Effective radius (3D half-stellar-mass radius) of a galaxy, given
    the halo virial radius and concentration, using the empirical formula
    of Jiang+19 (MN, 488, 4801) eq.6
    
        R_eff = 0.021 (c/10)^-0.7 R_vir
    
    Syntax:
    
        Reff(Rv,c2)
        
    where
    
        Rv: virial radius [kpc] (float or array)
        c2: halo concentration defined as R_vir / r_-2, where r_-2 is the
            radius at which dln(rho)/dln(r) = -2 (float or array)
    """
    return 0.021 * (c2/10.)**(-0.7) * Rv
    
#---stellar-halo-mass relation

def lgMs_B13(lgMv,z=0.):
    r"""
    Log stellar mass [M_sun] given log halo mass and redshift, using the 
    fitting function by Behroozi+13.
    
    Syntax:
    
        lgMs_B13(lgMv,z)
    
    where 
        lgMv: log virial mass [Msun] (float or array)
        z: redshift (float) (default=0.)
    """
    a = 1./(1.+z)
    v = v_B13(a)
    e0 = -1.777
    ea = -0.006
    ez = 0.000
    ea2 = -0.119
    M0 = 11.514
    Ma = -1.793
    Mz = -0.251
    lge = e0 + (ea*(a-1.)+ez*z)*v + ea2*(a-1.)
    lgM = M0 + (Ma*(a-1.)+Mz*z)*v
    return lge+lgM + f_B13(lgMv-lgM,a) - f_B13(0.,a)
def v_B13(a):
    r"""
    Auxiliary function for lgMs_B13.
    """
    return np.exp(-4.*a**2)
def f_B13(x,a):
    r"""
    Auxiliary function for lgMs_B13.
    """
    a0 = -1.412
    aa = 0.731
    az = 0.0
    d0 = 3.508
    da = 2.608
    dz = -0.043
    g0 = 0.316
    ga = 1.319
    gz = 0.279
    v = v_B13(a)
    z = 1./a-1.
    alpha = a0 + (aa*(a-1.)+az*z)*v
    delta = d0 + (da*(a-1.)+dz*z)*v
    gamma = g0 + (ga*(a-1.)+gz*z)*v
    return delta*(np.log10(1.+np.exp(x)))**gamma/(1.+np.exp(10**(-x)))-\
        np.log10(1.+10**(alpha*x))

def lgMs_RP17(lgMv,z=0.):
    """
    Log stellar mass [M_sun] given log halo mass and redshift, using the 
    fitting function by Rodriguez-Puebla+17.
    
    Syntax:
    
        lgMs_RP17(lgMv,z)
    
    where 
    
        lgMv: log virial mass [M_sun] (float or array)
        z: redshift (float) (default=0.)
    """
    a = 1./(1.+z)
    v = v_RP17(a)
    e0 = -1.758
    ea = 0.110
    ez = -0.061
    ea2 = -0.023
    M0 = 11.548
    Ma = -1.297
    Mz = -0.026
    lge = e0 + (ea*(a-1.)+ez*z)*v + ea2*(a-1.)
    lgM = M0 + (Ma*(a-1.)+Mz*z)*v
    return lge+lgM + f_RP17(lgMv-lgM,a) - f_RP17(0.,a)
def v_RP17(a):
    """
    Auxiliary function for lgMs_RP17.
    """
    return np.exp(-4.*a**2)
def f_RP17(x,a):
    r"""
    Auxiliary function for lgMs_RP17.
    
    Note that RP+17 use 10**( - alpha*x) while B+13 used 10**( +alpha*x).
    """
    a0 = 1.975
    aa = 0.714
    az = 0.042
    d0 = 3.390
    da = -0.472
    dz = -0.931
    g0 = 0.498
    ga = -0.157
    gz = 0.0
    v = v_RP17(a)
    z = 1./a-1.
    alpha = a0 + (aa*(a-1.)+az*z)*v
    delta = d0 + (da*(a-1.)+dz*z)*v
    gamma = g0 + (ga*(a-1.)+gz*z)*v
    return delta*(np.log10(1.+np.exp(x)))**gamma/(1.+np.exp(10**(-x)))-\
        np.log10(1.+10**( - alpha*x))

#---SMHR-inner-halo-density-slope relation

def slope(X):
    """
    Logarithmic halo density slope at 0.01 R_vir, as a function of the 
    stellar-to-halo-mass ratio X, based on the simulation results of 
    Di Cintio + (2014).
    
    Syntax:
    
        slope(X)
        
    where
    
        X: M_star / M_vir 
    """
    s = X / 10**(-2.051)
    #return np.log10( s**(-0.593) + s**1.99 ) - 0.132 # <<< fiducial
    return np.log10( 13.6 + s**1.99 ) - 0.132 # <<< test: no core
    
#---concentration-mass-redshift relations

def c2_Zhao09(Mv,t,version='zhao'):
    """
    Halo concentration from the mass assembly history, using the Zhao+09
    relation.
    
    Syntax:
    
        c2_Zhao09(Mv,t,type)
        
    where
    
        Mv: main-branch virial mass history [M_sun] (array)
        t: the time series of the main-branch mass history (array of the
            same size as Mv)
        version: 'zhao' or 'vdb' for the different versions of the
                 fitting function parameters (string)
    
    Note that we need Mv and t in reverse chronological order, i.e., in 
    decreasing order, such that Mv[0] and t[0] is the instantaneous halo
    mass and time.
    
    Note that Mv is the Bryan and Norman 98 M_vir.
    
    Return:
        
        halo concentration c R_vir / r_-2 (float)
    """
    if(version == 'vdb'):
        coeff1 = 3.40
        coeff2 = 6.5
    elif(version == 'zhao'):
        coeff1 = 3.75
        coeff2 = 8.4
    idx = aux.FindNearestIndex(Mv,0.04*Mv[0])
    return 4.*(1.+(t[0]/(coeff1*t[idx]))**coeff2)**0.125
    
def lgc2_DM14(Mv,z=0.):
    r"""
    Halo concentration given virial mass and redshift, using the 
    fitting formula from Dutton & Maccio 14 (eqs.10-11)
    
    Syntax:
    
        lgc2_DM14(Mv,z=0.)
    
    where 
    
        Mv: virial mass, M_200c [M_sun] (float or array)
        z: redshift (float or array of the same size as Mv,default=0.)
        
    Note that this is for M_200c, for the BN98 M_vir, use DM14 eqs.12-13
    instead. 
    
    Note that the parameters are for the Planck(2013) cosmology.
    
    Return:
    
        log of halo concentration c_-2 = R_200c / r_-2 (float or array)
    """
    # <<< concentration from NFW fit
    #a = 0.026*z - 0.101 # 
    #b = 0.520 + (0.905-0.520) * np.exp(-0.617* z**1.21)
    # <<< concentration from Einasto fit
    a = 0.029*z - 0.130
    b = 0.459 + (0.977-0.459) * np.exp(-0.490* z**1.303) 
    return a*np.log10(Mv*cfg.h/10**12.)+b