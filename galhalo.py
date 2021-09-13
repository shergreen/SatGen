################## Functions for galaxy-halo connection ####################

# Arthur Fangzhou Jiang 2019, HUJI
# Arthur Fangzhou Jiang 2020, Caltech
# Sheridan Beckwith Green 2020, Yale

#########################################################################

import numpy as np

import config as cfg
import aux
import profiles as pr
import cosmo as co

from lmfit import minimize, Parameters

#########################################################################

#---galaxy-size-halo-structure relation   

def Reff(Rv,c2):
    """
    Effective radius (3D half-stellar-mass radius) of a galaxy, given
    the halo virial radius and concentration, using the empirical formula
    of Jiang+19 (MN, 488, 4801) eq.6
    
        R_eff = 0.02 (c/10)^-0.7 R_vir
    
    Syntax:
    
        Reff(Rv,c2)
        
    where
    
        Rv: virial radius [kpc] (float or array)
        c2: halo concentration defined as R_vir / r_-2, where r_-2 is the
            radius at which dln(rho)/dln(r) = -2 (float or array)
    """
    return 0.02 * (c2/10.)**(-0.7) * Rv
    
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

#---halo-response patterns

def slope(X,choice='NIHAO'):
    """
    Logarithmic halo density slope at 0.01 R_vir, as a function of the 
    stellar-to-halo-mass ratio X, based on simulation results.
    
    Syntax:
    
        slope(X,choice='NIHAO')
        
    where
    
        X: M_star / M_vir (float or array)
        choice: choice of halo response -- 
            'NIHAO' (default, Tollet+16, mimicking strong core formation)
            'APOSTLE' (Bose+19, mimicking no core formation)
    """
    if choice=='NIHAO':
        s0 = X / 8.77e-3
        s1 = X / 9.44e-5
        return np.log10(26.49*(1.+s1)**(-0.85) + s0**1.66) + 0.158
    elif choice=='APOSTLE':
        s0 = X / 8.77e-3
        return np.log10( 20. + s0**1.66 ) + 0.158 

def c2c2DMO(X,choice='NIHAO'):
    """
    The ratio between the baryon-influenced concentration c_-2 and the 
    dark-matter-only c_-2, as a function of the stellar-to-halo-mass
    ratio, based on simulation results. 
    
    Syntax:
    
        c2c2DMO(X,choice='NIHAO')
        
    where
    
        X: M_star / M_vir (float or array)
        choice: choice of halo response -- 
            'NIHAO' (default, Tollet+16, mimicking strong core formation)
            'APOSTLE' (Bose+19, mimicking no core formation)
    """
    if choice=='NIHAO':
        #return 1. + 227.*X**1.45 - 0.567*X**0.131 # <<< Freundlich+20
        return 1.2 + 227.*X**1.45 - X**0.131 # <<< test
    elif choice=='APOSTLE':
        return 1. + 227.*X**1.45
        
#---concentration-mass-redshift relations

def c2_Zhao09(Mv,t,version='zhao'):
    """
    Halo concentration from the mass assembly history, using the Zhao+09
    relation.
    
    Syntax:
    
        c2_Zhao09(Mv,t,version)
        
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

def c2_DK15(Mv,z=0.,n=-2):
    """
    Halo concentration from Diemer & Kravtsov 15 (eq.9).
    
    Syntax:
    
        c2_DK15(Mv,z)
        
    where
    
        Mv: virial mass, M_200c [M_sun] (float or array)
        z: redshift (float or array of the same size as Mv,default=0.)
        n: logarithmic slope of power spectrum (default=-2 or -2.5 for
            typical values of LCDM, but more accurately, use the power
            spectrum to calculate n)
    
    Note that this is for M_200c.
    Note that the parameters are for the median relation
    
    Return:
        
        halo concentration R_200c / r_-2 (float)
    """
    cmin = 6.58 + 1.37*n
    vmin = 6.82 + 1.42*n
    v = co.nu(Mv,z,**cfg.cosmo)
    fac = v / vmin
    return 0.5*cmin*(fac**(-1.12)+fac**1.69)

#---halo contraction model

def contra_Hernquist(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial NFW halo profile (object of the NFW class as defined
            in profiles.py)
        d: baryon profile (object of the Hernquist class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be self-similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # prepare variables
    Mv = h.Mh
    c = h.ch
    rv = h.rh
    fc = h.f(c)
    Mb = d.Mb
    rb = d.r0
    fb = Mb/Mv
    xb = rb/rv
    x = r/rv
    xave = A * x**w
    rave = xave * rv # orbit-averaged radii
    # compute y_0
    a = 2.*fb*(1.+xb)**2 * fc / (xb*c)**2
    fdm = 1.-fb
    s = 0.5/a 
    p = 1.+2.*w
    sqrtQ1 = np.sqrt( (fdm/(3.*a))**3 + s**2 )
    sqrtQw = np.sqrt( (fdm/p)**p / a**3 + s**2 )
    y1 = (sqrtQ1 + s)**(1./3.) - (sqrtQ1 - s)**(1./3.)
    yw = (sqrtQw + s)**(1./p) - (sqrtQw - s)**(1./p)
    em2a = np.exp(-2.*a) 
    y0 = y1*em2a + yw*(1.-em2a)
    # compute exponent b
    b = 2.*y0/(1.-y0)*(2./xb-4.*c/3.)/(2.6+fdm/(a*y0**(2.*w)))
    # compute the contraction ratio y(x)=r_f / r
    Mi = h.M(rave)
    t0 = 1./(fdm + d.M(y0**w *rave)/Mi)
    t1 = 1./(fdm + d.M(rave)/Mi)
    embx = np.exp(-b*x)
    y = t0*embx + t1*(1.-embx)
    rf = y*r
    return rf, fdm*h.M(r)
    
def contra_exp(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Similar to "contra_Hernquist", but here we assume the final baryon
    distribution to be an exponential disk, instead of a spherical 
    Hernquist profile
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial NFW halo profile (object of the NFW class as defined
            in profiles.py)
        d: baryon profile (object of the exponential class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be self-similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # prepare variables
    Mv = h.Mh
    c = h.ch
    rv = h.rh
    fc = h.f(c)
    Mb = d.Mb
    rb = d.r0
    fb = Mb/Mv
    xb = rb/rv
    x = r/rv
    xave = A * x**w
    rave = xave * rv # orbit-averaged radii
    # compute y_0
    a = fb * fc / (xb*c)**2
    fdm = 1.-fb
    s = 0.5/a 
    p = 1.+2.*w
    sqrtQ1 = np.sqrt( (fdm/(3.*a))**3 + s**2 )
    sqrtQw = np.sqrt( (fdm/p)**p / a**3 + s**2 )
    y1 = (sqrtQ1 + s)**(1./3.) - (sqrtQ1 - s)**(1./3.)
    yw = (sqrtQw + s)**(1./p) - (sqrtQw - s)**(1./p)
    em2a = np.exp(-2.*a) 
    y0 = y1*em2a + yw*(1.-em2a)
    # compute exponent b
    b = 2.*y0/(1.-y0)*(2./(3.*xb)-4.*c/3.)/(2.6+fdm/(a*y0**(2.*w)))
    # compute the contraction ratio y(x)=r_f / r
    Mi = h.M(rave)
    t0 = 1./(fdm + d.M(y0**w *rave)/Mi)
    t1 = 1./(fdm + d.M(rave)/Mi)
    embx = np.exp(-b*x)
    y = t0*embx + t1*(1.-embx)
    rf = y*r
    return rf, fdm*h.M(r)
    
def contra(r,h,d,A=0.85,w=0.8):
    """
    Returns contracted halo profile given baryon profile and initial halo 
    profile, following the model of Gnedin+04.
    
    Syntax:
    
        contra(r,h,d)
        
    where
    
        r: initial radii at which we evaluate the mass profile [kpc]
            (array)
        h: initial NFW halo profile (object of the NFW class as defined
            in profiles.py)
        d: baryon profile (object of the Hernquist class as defined in
            profiles.py)
        A: coefficient in the relation between the orbit-averaged radius 
            of a particle that is currently in a shell and the instant
            radius of the shell: <r>/r_vir = A (r/r_vir)^w 
            (default=0.85)
        w: power-law index in the relation between the orbit-averaged
            radius and instant radius (default=0.8)
            
    Note that there is halo-to-halo variation in the values of A and w,
    which is discussed in Gnedin+11. Here we ignore the halo-to-halo 
    variation and adopt the fiducial values A=0.85 and w=0.8 as in 
    Gnedin+04.
    
    Note that the input halo object "h" is for the total mass profile,
    which includes an initial baryon mass distribution that is assumed
    to be self-similar to the initial DM profile, i.e.,
    
        M_dm,i = (1-f_b) M_i(r)
        M_b,i = f_b M_i(r)
            
    Return:
    
        the contracted DM profile (object of the Dekel class as defined
            in profiles.py) 
        contracted radii, r_f [kpc] (array of the same length as r)
        enclosed DM mass at r_f [M_sun] (array of the same length as r) 
    """
    # contract
    if isinstance(d,pr.Hernquist):
        rf,Mdmf = contra_Hernquist(r,h,d,A,w)
    elif isinstance(d,pr.exp):
        rf,Mdmf = contra_exp(r,h,d,A,w)
    # fit contracted profile
    params = Parameters()
    params.add('Mv', value=(1.-d.Mb/h.Mh)*h.Mh, vary=False)
    params.add('c', value=h.ch,min=1.,max=100.)
    params.add('a', value=1.,min=-2.,max=2.)
    out = minimize(fobj_Dekel, params, args=(rf,Mdmf,h.Deltah,h.z)) 
    MvD = out.params['Mv'].value
    cD = out.params['c'].value
    aD = out.params['a'].value
    return pr.Dekel(MvD,cD,aD),rf,Mdmf 
def fobj_Dekel(p, xdata, ydata, Delta, z):
    """
    Auxiliary function for "contra" -- objective function for fitting
    a Dekel+ profile to the contracted halo
    """
    h = pr.Dekel(p['Mv'].value,p['c'].value,p['a'].value,Delta=Delta,z=z)
    ymodel = h.M(xdata)
    return (ydata - ymodel) / ydata
