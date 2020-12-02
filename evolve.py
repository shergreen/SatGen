################## Functions for satellite evolution ####################

# Arthur Fangzhou Jiang 2016, HUJI --- original version
# Arthur Fangzhou Jiang 2019, HUJI, UCSC --- revisions
# Sheridan Beckwith Green 2020, Yale University

#########################################################################

import config as cfg
import cosmo as co
import profiles as pr

import numpy as np
from scipy.interpolate import interp1d,interp2d
from scipy.optimize import brentq

#########################################################################

#---tidal tracks    

alpha_grid_P10 = np.array([1.5,1.,0.5,0.])
mu_vmax_grid_P10 = np.array([0.4,0.4,0.4,0.4])
eta_vmax_grid_P10 = np.array([0.24,0.3,0.35,0.37])
mu_rmax_grid_P10 = np.array([0.,-0.3,-0.4,-1.3])
eta_rmax_grid_P10 = np.array([0.48,0.4,0.27,0.05])
mu_vmax_interp_P10 = interp1d(alpha_grid_P10,mu_vmax_grid_P10,
    kind='cubic')
eta_vmax_interp_P10 = interp1d(alpha_grid_P10,eta_vmax_grid_P10,
    kind='cubic')
mu_rmax_interp_P10 = interp1d(alpha_grid_P10,mu_rmax_grid_P10,
    kind='cubic')
eta_rmax_interp_P10 = interp1d(alpha_grid_P10,eta_rmax_grid_P10,
    kind='cubic')
def g_P10(x,alpha=1.):
    """
    Penarrubia+10 tidal tracks, i.e., the evolution of v_max(t)/v_max(0)
    and l_max(t)/l_max(0) as functions of the bound mass fraction 
    m(t)/m(0)

    Syntax:
    
        g_P10(x,alpha=1.)
    
    where
    
        x:=m(t)/m(0), i.e., the bound mass fraction (float)
        alpha: the initial inner slope of a subhalo, i.e., the initial 
            "gamma" in the Zhao96 and Kravtsov+98 alpha-beta-gamma 
            profile (default=1., appropriate for NFW)
            
    Return:
    
        v_max(t)/v_max(0), l_max(t)/l_max(0)
    """
    if alpha < alpha_grid_P10.min():
        alpha = alpha_grid_P10.min()
    if alpha > alpha_grid_P10.max():
        alpha = alpha_grid_P10.max()
    mu_vmax = mu_vmax_interp_P10(alpha)
    eta_vmax = eta_vmax_interp_P10(alpha)
    mu_rmax = mu_rmax_interp_P10(alpha)
    eta_rmax = eta_rmax_interp_P10(alpha)
    y = 2./(1.+x)
    return y**mu_vmax * x**eta_vmax, y**mu_rmax * x**eta_rmax
    
alpha_grid_EPW18 = np.array([1.5,1.,0.5,0.])
lefflmax_grid_EPW18 = np.array([0.05,0.1])
alpha_mesh_EPW18,lefflmax_mesh_EPW18 = np.meshgrid(alpha_grid_EPW18,
    lefflmax_grid_EPW18)
# lgxs_leff_mesh_EPW18 = np.array([[-3.96,-2.64,-1.32,0.],
#     [-3.12,-2.08,-1.04,0.]])
# mu_leff_mesh_EPW18 = np.array([[0.83,0.47,0.11,-0.25],
#     [0.865,0.5,0.135,-0.23]])
# eta_leff_mesh_EPW18 = np.array([[0.74,0.41,0.08,-0.25],
#     [0.745,0.42,0.095,-0.23]])
# lgxs_mstar_mesh_EPW18 = np.array([[-2.4,-2.64,-2.88,-3.12],
#     [-1.955,-2.08,-2.205,-2.33]])
# mu_mstar_mesh_EPW18 = np.array([[1.39,1.87,2.35,2.83],
#     [1.675,1.8,1.925,2.05]])
# eta_mstar_mesh_EPW18 = np.array([[1.39,1.87,2.35,2.83],
#     [1.675,1.8,1.925,2.05]])
lgxs_leff_mesh_EPW18 = np.array([[-2.4,-2.64,-2.9,-3.12],
    [-2.,-2.08,-2.2,-2.33]])
mu_leff_mesh_EPW18 = np.array([[0.59,0.47,0.19,-0.15],
    [0.75,0.5,0.21,-0.15]])
eta_leff_mesh_EPW18 = np.array([[0.59,0.41,0.07,-0.35],
    [0.71,0.42,0.09,-0.33]])
lgxs_mstar_mesh_EPW18 = np.array([[-2.4,-2.64,-2.9,-3.12],
    [-2.,-2.08,-2.2,-2.33]])
mu_mstar_mesh_EPW18 = np.array([[1.39,1.87,2.35,2.83],
    [1.68,1.8,1.93,2.05]])
eta_mstar_mesh_EPW18 = np.array([[1.39,1.87,2.35,2.83],
    [1.68,1.8,1.93,2.05]])
lgxs_leff_interp_EPW18 = interp2d(alpha_grid_EPW18,lefflmax_grid_EPW18,
    lgxs_leff_mesh_EPW18,kind='linear')
mu_leff_interp_EPW18 = interp2d(alpha_grid_EPW18,lefflmax_grid_EPW18,
    mu_leff_mesh_EPW18,kind='linear')
eta_leff_interp_EPW18 = interp2d(alpha_grid_EPW18,lefflmax_grid_EPW18,
    eta_leff_mesh_EPW18,kind='linear')
lgxs_mstar_interp_EPW18 = interp2d(alpha_grid_EPW18,lefflmax_grid_EPW18,
    lgxs_mstar_mesh_EPW18,kind='linear')
mu_mstar_interp_EPW18 = interp2d(alpha_grid_EPW18,lefflmax_grid_EPW18,
    mu_mstar_mesh_EPW18,kind='linear')
eta_mstar_interp_EPW18 = interp2d(alpha_grid_EPW18,lefflmax_grid_EPW18,
    eta_mstar_mesh_EPW18,kind='linear')
def g_EPW18(x,alpha=1.,lefflmax=0.1):
    """
    Errani, Penerrubia, & Walker (2018) tidal tracks, i.e., the evolution
    of l_eff(t)/l_eff(0) and m_star(t)/m_star(0) as functions of the mass
    within l_max, i.e., m_max(t)/m_max(0)
    
    Syntax:
    
        g_EPW18(x,alpha=1.,lefflmax=0.1)
    
    where
    
        x:=m_max(t)/m_max(0), i.e., the mass within l_max wrt the initial 
            value of that (float)
        alpha: the initial inner slope of a subhalo, i.e., the initial 
            "gamma" in the Zhao96 and Kravtsov+98 alpha-beta-gamma 
            profile (default=1., appropriate for NFW)
        lefflmax: the initial ratio between l_eff and l_max, i.e., how
            segregated the stars and DM are initially (default=0.1)
            
    Return:
    
        l_eff(t)/l_eff(0), m_star(t)/m_star(0)
    """
    xs_leff = 10.**lgxs_leff_interp_EPW18(alpha,lefflmax)
    mu_leff = mu_leff_interp_EPW18(alpha,lefflmax)
    eta_leff = eta_leff_interp_EPW18(alpha,lefflmax)
    xs_mstar = 10.**lgxs_mstar_interp_EPW18(alpha,lefflmax)
    mu_mstar = mu_mstar_interp_EPW18(alpha,lefflmax)
    eta_mstar = eta_mstar_interp_EPW18(alpha,lefflmax)
    y_leff = (1.+xs_leff)/(x+xs_leff)
    y_mstar = (1.+xs_mstar)/(x+xs_mstar)
    return y_leff**mu_leff *x**eta_leff, y_mstar**mu_mstar *x**eta_mstar
 
def Dekel(mv,mv0,lmax0,vmax0,alpha0,z=0.):
    """
    Use the Penarrubia+10 tidal tracks to evolve a satellite described by
    a Dekel+17 profile, assuming that the innermost slope alpha doesn't
    change.
    
    Syntax:
    
        Dekel(mv,mv0,lmax0,vmax0,alpha0,z=0.)
    
    where
    
        mv: the evolved virial mass [M_sun] (float or array)
        mv0: the initial virial mass [M_sun] (float) 
        lmax0: the initial l_max, i.e., the radius where the maximum 
            circular velocity is reached [kpc] (float)
        vmax0: the initial v_max, i.e., the maximum circular velocity
            [kpc/Gyr] (float)
        alpha0: the initial innermost logarithmic density slope (float)
        z: redshift, for computing the critical density rho_crit (float)
            (default=0.)
        
    Return:
    
        the new Dekel concentration, c(float), 
        the new spherical overdensity, Delta (float)
    """
    g_vmax,g_lmax = g_P10(mv/mv0,alpha0)
    lmax = lmax0 * g_lmax
    vmax = vmax0 * g_vmax
    s2 = 2.-alpha0
    s3 = 3.-alpha0
    A = (cfg.G * mv / lmax / vmax**2)**(0.5/s3) * (s2/s3)
    lv = lmax / s2**2 * A**2 / (1.-A)**2
    c = s2**2 * lv / lmax
    rhoc = co.rhoc(z,h=cfg.h,Om=cfg.Om,OL=cfg.OL)
    Delta = 3.*mv / (cfg.FourPi * lv**3 * rhoc)
    return c,Delta 
    
def Dekel2(mv,mv0,lmax0,vmax0,alpha0,slope0,z=0.):
    """
    Use the Penarrubia+10 tidal tracks to evolve a satellite described by
    a Dekel+17 profile, assuming that the innermost slope alpha doesn't
    change.
    
    Note that this is a variant of the "Dekel" function: use the initial
    inner slope at l ~ 0.01 l_vir, instead of the innermost slope, as the 
    slope condition for the tidal tracks.
    
    Syntax:
    
        Dekel2(mv,mv0,lmax0,vmax0,alpha0,slope0,z=0.)
    
    where
    
        mv: the evolved virial mass [M_sun] (float or array)
        mv0: the initial virial mass [M_sun] (float) 
        lmax0: the initial l_max, i.e., the radius where the maximum 
            circular velocity is reached [kpc] (float)
        vmax0: the initial v_max, i.e., the maximum circular velocity
            [kpc/Gyr] (float)
        alpha0: the initial innermost logarithmic density slope (float)
        slope0: the initial inner logarithmic density slope at 
            l ~ 0.01 l_vir -- this is used as the slope condition for the
            tidal tracks (float)
        z: redshift, for computing the critical density rho_crit (float)
            (default=0.)
        
    Return:
    
        the new Dekel concentration, c(float), 
        the new spherical overdensity, Delta (float)
    """
    g_vmax,g_lmax = g_P10(mv/mv0,slope0)
    lmax = lmax0 * g_lmax
    vmax = vmax0 * g_vmax
    s2 = 2.-alpha0
    s3 = 3.-alpha0
    A = (cfg.G * mv / lmax / vmax**2)**(0.5/s3) * (s2/s3)
    lv = lmax / s2**2 * A**2 / (1.-A)**2
    c = s2**2 * lv / lmax
    rhoc = co.rhoc(z,h=cfg.h,Om=cfg.Om,OL=cfg.OL)
    Delta = 3.*mv / (cfg.FourPi * lv**3 * rhoc)
    return c,Delta 

#---for tidal stripping

def alpha_from_c2(c2p, c2s):
    """
    Compute the best stripping efficiency prefactor, alpha, as a function of
    the instantaneous host- and initial subhalo concentrations.

    Syntax:
    
        alpha_from_c2(c2p, cs2)

    where

        c2p: instantaneous host NFW concentration
        c2s: initial subhalo NFW concentration

    Return

        stripping efficiency, alpha (float)

    NOTE:

        The initial NFW subhalo concentration is used, as the
        Green and van den Bosch (2019) density profile model takes
        into account the evolution of the density profile of the subhalo
        given the INITIAL NFW profile.

    TODO:
    
        Update this based on the final DASH calibration.
    """
    return 0.55 * ((c2s/c2p) / 2.)**(-1./3.)


def msub(sp,potential,xv,dt,choice='King62',alpha=1.):
    """
    Evolve subhalo mass due to tidal stripping, by an amount of
    
        alpha * [m - m(l_t)] * dt/t_dyn
        
    where 
    
        m is the satellite virial mass; 
        m(l_t) is the satellite mass within the tidal radius l_t; 
        dt is the time step; 
        t_dyn is the host dynamical locally at the satellite's position.
    
    Syntax:
    
        mass(sp,potential,xv,dt,choice='King62',alpha=1.)
    
    where
        
        sp: satellite potential (an object of one of the classes defined 
            in profiles.py)
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
        dt: time interval [Gyr] (float)
        choice: choice of tidal radius expression, including
            "King62" (default): eq.(12.21) of Mo, van den Bosch, White 10
            "Tormen98": eq.(3) of van den Bosch+17
        alpha: stripping efficienty parameter -- the larger the 
            more effient (default=1.)
            
    Return
    
        evolved mass, m [M_sun] (float)
        tidal radius, l_t [kpc] (float)
    """
    lt = ltidal(sp,potential,xv,choice)
    if lt<sp.rh: 
        dm = alpha * (sp.Mh-sp.M(lt)) * dt/pr.tdyn(potential,xv[0],xv[2])
        dm = max(dm,0.) # avoid negative dm
        m = max(sp.Mh-dm,cfg.fbv_min*sp.Minit) 
        # changed from max(sp.Mh-dm,cfg.Mres), which is used for fixed Mres
    else:
        m = sp.Mh
    return m,lt
def ltidal(sp,potential,xv,choice='King62'):
    """
    Tidal radius [kpc] of a satellite, given satellite profile, host
    potential, and phase-space coordinate within the host. 
    
    Syntax:
    
        ltidal(sp,potential,xv,choice='King62')
    
    where
    
        sp: satellite potential (an object define in profiles.py)
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
        choice: choice of tidal radius expression, including
            "King62" (default): eq.(12.21) of Mo, van den Bosch, White 10
            "Tormen98": eq.(3) of van den Bosch+18
            
    Note that the only difference between King62 and Tormen98 is that 
    the latter ignores the centrifugal force and thus gives larger tidal 
    radius (i.e., weaker tidal stripping). 
    """
    a = cfg.Rres
    b = 9.999*sp.rh
    if choice=='King62':
        rhs = lt_King62_RHS(potential,xv)
    elif choice=='Tormen98':
        rhs = lt_Tormen98_RHS(potential,xv)
    else: 
        sys.exit('Invalid choice of tidal radius type!')

    fa = Findlt(a,sp,rhs)
    fb = Findlt(b,sp,rhs)
    if fa*fb>0.:
        lt = cfg.Rres
    else:
        lt = brentq(Findlt, a,b, args=(sp,rhs),
            rtol=1e-5,maxiter=1000)
    return lt
def lt_Tormen98_RHS(potential,xv):
    """
    Auxiliary function for 'ltidal', which returns the right-hand side
    of the Tormen98 equation for tidal radius, as in eq.(3) of 
    van den Bosch+18, but inverted and with all subhalo terms on
    left-hand side.
    
    Syntax:
    
        lt_Tormen98_RHS(potential,xv)
    
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
    """
    r = np.sqrt(xv[0]**2.+xv[2]**2.)
    M = pr.M(potential,r)
    rho = pr.rho(potential,r)
    dlnMdlnr = cfg.FourPi * r**3 * rho / M
    return (M / r**3) * (2. - dlnMdlnr)
def lt_King62_RHS(potential,xv):
    """
    Auxiliary function for 'ltidal', which returns the right-hand side
    of the King62 equation for tidal radius, as in eq.(12.21) of 
    Mo, van den Bosch, White 10, but inverted and with all subhalo
    terms on left-hand side.
    
    Syntax:
    
        lt_King62_RHS(potential,xv)
    
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
    """
    r = np.sqrt(xv[0]**2.+xv[2]**2.)
    Om = Omega(xv)
    M = pr.M(potential,r)
    rho = pr.rho(potential,r)
    dlnMdlnr = cfg.FourPi * r**3 * rho / M
    return (M / r**3) * (2.+Om**2.*r**3/cfg.G/M - dlnMdlnr)
def Findlt(l,sp,rhs):
    """
    Auxiliary function for 'ltidal', which returns the 
    
        left-hand side - right-hand side
    
    of the equation for tidal radius. Note that this works
    for either the Tormen98 or King62, since all differences
    are contained in the pre-computed right-hand side.
    
    Syntax:
    
        Findlt(l,sp,rhs)
    
    where
    
        l: radius in the satellite [kpc] (float)
        sp: satellite potential (an object define in profiles.py)
        rhs: right-hand side of equation, computed by either
        lt_Tormen98_RHS() or lt_King62_RHS() (float)
    """
    m = sp.M(l)
    return (m / l**3) - rhs
def Omega(xv):
    """
    Angular speed [Gyr^-1] upon input of phase-space coordinates
    
    Syntax: 
    
        Omega(xv):
    
    where
    
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
    """
    rsqr = xv[0]**2.+xv[2]**2.
    rxv = np.cross( np.array([xv[0],0.,xv[2]]) , xv[3:6] )
    return np.sqrt(rxv[0]**2.+rxv[1]**2.+rxv[2]**2.) / rsqr
    
def mgas(sg,sp,gpotential,potential,xv,dt,kappa=1.,alpha=1.):
    """
    Evolve satellite gas mass due to tidal stripping, by an amount of
    
        [m - m(l_rp)] * dt / t_dyn
        
    where m is the satellite gas mass; m(l_rp) is the satellite gas mass 
    within ram pressure radius l_rp; dt is the timestep size; and t_dyn 
    is the host dynamical time within radius r. 
    ( r is given by np.sqrt(xv[0]**2.+xv[2]**2.) )
    
    Syntax:
    
        mgas(sg,sp,gpotential,potential,xv,dt,kappa=1.,alpha=1.)
    
    where
    
        sg: satellite gas profile (an object of one of the classes 
            defined in profiles.py)
        sp: satellite potential (an object of one of the classes defined 
            in profiles.py)
        gpotential: the gas part of the host potential (a density profile 
            object, or a list of such objects that constitute a composite 
            potential)
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
        dt: time interval [Gyr] (float)
        kappa: the fudge factor of order unity in front of the 
            gravitational restoring pressure (0.5-2 depending on 
            assumptions, see Zinger+18 for details) (default=1.)
        alpha: stripping efficienty parameter -- the larger the 
            more effient (default=1.)

    Return:
    
        evolved gas mass, m [M_sun] (float)
        ram-pressure radius, l_rp [kpc] (float)
    """
    lrp = lram(sg,sp,gpotential,xv,kappa)
    if lrp<sg.rh: 
        dm = alpha *(sg.Mh-sg.M(lrp)) *dt/pr.tdyn(potential,xv[0],xv[2])
        m = max(sg.Mh-dm,cfg.Mres)
    else:
        m = sg.Mh
    return m,lrp
def lram(sg,sp,gpotential,xv,kappa=1.):
    r"""
    Ram-pressure radius [kpc] of a satellite, given satellite gas 
    profile, satellite profile, host halo gas profile, and phase space 
    coordinate within the host.  
    
    Syntax:
    
        lram(sg,sp,gpotential,xv,kappa=1.)
    
    where
    
        sg: satellite gas profile (an object of one of the classes 
            defined in profiles.py)
        sp: satellite potential (an object of one of the classes defined 
            in profiles.py)
        gpotential: the gas part of the host potential (a density profile 
            object, or a list of such objects that constitute a composite 
            potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
        kappa: the fudge factor of order unity in front of the 
            gravitational restoring pressure (0.5-2 depending on 
            assumptions, see Zinger+18 for details) (default=1.)
    """
    a = cfg.Rres
    b = 10.*sg.rh
    fa = Findlrp(a,sg,sp,gpotential,xv,kappa)
    fb = Findlrp(b,sg,sp,gpotential,xv,kappa)
    if fa*fb>0.:
        lrp = cfg.Rres
    else:
        lrp = brentq(Findlrp, a,b, args=(sg,sp,gpotential,xv,kappa),
            rtol=1e-5,maxiter=1000)
    return lrp
def Findlrp(l,sg,sp,gpotential,xv,kappa):
    r"""
    Auxiliary function for "lram", returns the
     
        left-hand side - right-hand side
    
    of the equation for ram pressure stripping radius:
    
        kappa * G m(l) rho_sat_gas(l) / l = rho_host_gas(r) v(r)^2
    
    Syntax:
   
        Findlrp(l,sg,sp,gpotential,xv,kappa)
   
    where: 
    
        l: radius in the satellite [kpc] (float)
        sg: satellite gas profile (an object of one of the classes 
            defined in profiles.py)
        sp: satellite potential (an object of one of the classes 
            defined in profiles.py)
        gpotential: the gas part of the host potential (a density profile 
            object, or a list of such objects that constitute a composite 
            potential)
        xv: phase-space coordinates [R,phi,z,VR,Vphi,Vz] in units of 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] (float array)
        kappa: the fudge factor in front of the gravitational restoring 
            pressure, that is of order unity (0.5-2 depending on 
            assumptions, see Zinger+18 for details) (default=1.)
    """
    V = np.sqrt(xv[3]**2. + xv[4]**2. + xv[5]**2.)
    rho = pr.rho(gpotential,xv[0],xv[2])
    return  kappa*cfg.G*sp.M(l)*sg.rho(l)/l - rho*V**2
