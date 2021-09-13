####################### potential well classes ##########################

# Arthur Fangzhou Jiang (2016, HUJI) --- original version

# Arthur Fangzhou Jiang (2019, HUJI and UCSC) --- revisions
# 
# - Dekel+ profile added some time earlier than 2019
# - support of velocity dispersion profile and thus dynamical friction 
#   (DF) for Dekel+ profile 
# - improvement of DF implementation: 
#   1. now compute the common part of the Chandrasekhar formula only once
#      for each profile class (as opposed to three times in each of the 
#      force component .FR, .Fphi, and .Fz);
#   2. removed satelite-profile-dependent CoulombLogChoice, and now 
#      CoulombLogChoices only depend on satellite mass m and host 
#      potential 

# Sheridan Beckwith Green (2020, Yale University) --- revisions
#
# - Added Green profile, which is an NFW profile multiplied by the
#   transfer function from Green and van den Bosch (2019)

#########################################################################

import config as cfg # for global variables
import cosmo as co # for cosmology related functions
import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.integrate import quad,odeint
from scipy.special import erf,gamma,gammainc,gammaincc 

#---variables for NFW mass definition change interpolation
x_interpolator = None
x_interpolator_min = None
x_interpolator_max = None

def gamma_lower(a,x):
    """
    Non-normalized lower incomplete gamma function
    
        integrate t^(a-1) exp(-t) from t=0 to t=x
        
    Syntax:
        
        gamma_lower(a,x)
    """
    return gamma(a)*gammainc(a,x)
def gamma_upper(a,x):
    """
    Non-normalized upper incomplete gamma function
    
        integrate t^(a-1) exp(-t) from t=x to t=infty
        
    Syntax:
        
        gamma_upper(a,x)
    """
    return gamma(a)*gammaincc(a,x)

#########################################################################

#---
class NFW(object):
    """
    Class that implements the Navarro, Frenk, & White (1997) profile:

        rho(R,z) = rho_crit * delta_char / [(r/r_s) * (1+r/r_s)^2]
                 = rho_0 / [(r/r_s) * (1+r/r_s)^2]
    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        r_s: scale radius, at which d ln rho(r) / d ln(r) = -2
        rho_crit: critical density of the Universe
        delta_char = Delta_halo / 3 * c^3 / f(c), where c = R_vir / r_s 
            is the concentration parameter
    
    Syntax:
    
        halo = NFW(M,c,Delta=200.,z=0.,sf=1.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: halo concentration (float)
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
        sf: Suppression factor used for reducing the overall
                density of the halo while preserving its shape, used
                when a disk is added in order to preserve total mass
                of the system
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .Vmax: maximum circular velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)      
        .d2Phidr2(R,z=0.): second radial derivative of potential [1/Gyr^2]
            at radius r=sqrt(R^2+r^2)
    
    HISTORY: Arthur Fangzhou Jiang (2016-10-24, HUJI)
             Arthur Fangzhou Jiang (2016-10-30, HUJI)
             Arthur Fangzhou Jiang (2019-08-24, HUJI)
    """
    def __init__(self,M,c,Delta=200.,z=0.,sf=1.):
        """
        Initialize NFW profile.
        
        Syntax:
        
            halo = NFW(M,c,Delta=200.,z=0.,sf=1.)
        
        where
        
            M: halo mass [M_sun] (float), 
            c: halo concentration (float),        
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default is 200.)         
            z: redshift (float)
            sf: Suppression factor used for reducing the overall
                density of the halo while preserving its shape, used
                when a disk is added in order to preserve total mass
                of the system
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.Deltah = Delta
        self.z = z
        self.sf = sf
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rmax = self.rs * 2.163
        self.rho0 = self.sf*self.rhoc*self.Deltah/3.*self.ch**3./self.f(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2.      
        self.Vmax = self.Vcirc(self.rmax)
        self.s001 = self.s(0.01*self.rh)
    def f(self,x):
        """
        Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)
    
        Syntax:
    
            .f(x)
        
        where
        
            x: dimensionless radius r/r_s (float or array)
        """
        return np.log(1.+x) - x/(1.+x) 
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.rho0 / (x * (1.+x)**2.)
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return 1. + 2*x / (1.+x)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return cfg.FourPi*self.rho0*self.rs**3. * self.f(x)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
        
        where 
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)   
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return 3.*self.rho0 * self.f(x)/x**3.
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).

        Syntax:
        
            .tdyn(R,z=0.)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)     
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return self.Phi0 * np.log(1.+x)/x
    def otherMassDefinition(self,Delta=200.):
        """
        Computes the mass, radius, and concentration of the fixed,
        physical halo under a new spherical overdensity definition.
        Since rho0 is fixed, determines the cnew=rnew/rs that solves:

            rho0 = [Delta * rhoc / 3] * (rnew/rs)**3 / f(rnew/rs)**3

        Implementation based on Benedikt Diemer's COLOSSUS code.

        Syntax:

            .otherMassDefinition(Delta=200.)

        where
            Delta: Spherical overdensity in units of the critical
                   overdensity at the redshift that the halo was
                   initialized at (float).

        Return:

            Mnew: Mass within new overdensity (Msun, float)
            rnew: Radius corresponding to new overdensity (kpc, float)
            cnew: Concentration relative to new overdensity.
        """

        global x_interpolator
        global x_interpolator_min
        global x_interpolator_max

        if x_interpolator is None:
            table_x = np.logspace(4.0, -4.0, 1000)
            table_y = self.f(table_x) * 3.0 / table_x**3
            x_interpolator = InterpolatedUnivariateSpline(table_y, 
                                                          table_x, k=3)
            knots = x_interpolator.get_knots()
            x_interpolator_min = knots[0]
            x_interpolator_max = knots[-1]

        dens_threshold = Delta * self.rhoc
        y = dens_threshold / self.rho0

        if(y < x_interpolator_min):
            raise Exception("Requested overdensity %.2e cannot be evaluated\
                             for scale density %.2e, out of range." \
                             % (y, x_interpolator_min))
        elif(y > x_interpolator_max):
            raise Exception("Requested overdensity %.2e cannot be evaluated\
                             for scale density %.2e, out of range." \
                             % (y, x_interpolator_max))

        cnew = x_interpolator(y)
        rnew = cnew * self.rs
        Mnew = self.M(rnew)
        return Mnew, rnew, cnew
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs   
        fac = self.Phi0 * (self.f(x)/x) / r**2.
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] at radius r = sqrt(R^2 + z^2), 
        assuming isotropic velicity dispersion tensor, and following the 
        Zentner & Bullock (2003) fitting function:
        
            sigma(x) = V_max 1.4393 x^0.345 / (1 + 1.1756 x^0.725)
            
        where x = r/r_s.
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
       return self.Vmax*1.4393*x**0.354/(1.+1.1756*x**0.725)
    def sigma_accurate(self,R,z=0.,beta=0.):
        """
        Velocity dispersion [kpc/Gyr].
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
            beta: anisotropy parameter (default=0., i.e., isotropic)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(self.dIdx_sigma, xx, np.inf,args=(beta,))[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf,args=(beta,))[0]
        f = self.f(x)
        sigmasqr = -self.Phi0 / x**(2.*beta-1) *(1.+x)**2 * I
        return np.sqrt(sigmasqr)
    def dIdx_sigma(self,x,beta):
        """
        Integrand for the integral in the velocity dispersion.
        """
        f = self.f(x)
        return x**(2.*beta-3.) * f / (1.+x)**2
    def dlnsigmasqrdlnr_accurate(self,R,z=0.,beta=0.):
        """
        d ln sigma^2 / d ln r
        """
        r = np.sqrt(R**2.+z**2.)
        r1 = r * (1.+cfg.eps)
        r2 = r * (1.-cfg.eps)
        y1 = np.log(self.sigma_accurate(r1))
        y2 = np.log(self.sigma_accurate(r2))
        return (y1-y2)/(r1-r2)
    def d2Phidr2(self,R,z=0.):
        """
        Second radial derivative of the gravitational potential [1/Gyr^2] 
        computed at r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .d2Phidr2(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs

        f = (2.*np.log(1. + x) - x*(2. + 3.*x)/(1. + x)**2.) / r**3.
        return self.Phi0 * self.rs * f
        
class Burkert(object):
    """
    Class that implements the Burkert (1995) profile:

        rho(R,z) = rho_0 / [(1+x)(1+x^2)], x = r/r_s
    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        r_s: scale radius, at which d ln rho(r) / d ln(r) = -2
        rho_crit: critical density of the Universe
        delta_char = Delta_halo / 3 * c^3 / f(c), where c = R_vir / r_s 
            is the concentration parameter
    
    Syntax:
    
        halo = Burkert(M,c,Delta=200.,z=0.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: Burkert concentration, R_vir/r_s (float)
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .rho0: central density [M_sun kpc^-3]
        .Vmax: maximum circualr velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)      
    
    HISTORY: Arthur Fangzhou Jiang (2020-07-29, Caltech)
    """
    def __init__(self,M,c,Delta=200.,z=0.):
        """
        Initialize Burkert profile.
        
        Syntax:
        
            halo = Burkert(M,c,Delta=200.,z=0.)
        
        where
        
            M: halo mass [M_sun] (float), 
            c: halo concentration (float),        
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default is 200.)         
            z: redshift (float)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rho0=self.Mh/cfg.TwoPi/self.rh**3/self.f(self.ch)*self.ch**3
        self.rmax = 3.24 * self.rs
        self.Vmax = self.Vcirc(self.rmax)    
        self.s001 = self.s(0.01*self.rh)
    def f(self,x):
        """
        Auxiliary method for NFW profile: 
            
            f(x) = 0.5 ln(1+x^2) + ln(1+x) - arctan(x)
    
        Syntax:
    
            .f(x)
        
        where
        
            x: dimensionless radius r/r_s (float or array)
        """
        return 0.5*np.log(1.+x**2)+np.log(1.+x)-np.arctan(x)
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.rho0 / ((1.+x) * (1.+x**2))
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        xsqr = x**2
        return x / (1.+x) + 2*xsqr / (1.+xsqr)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return cfg.TwoPi*self.rho0*self.rs**3 * self.f(x)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
        
        where 
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)   
        """
        r = np.sqrt(R**2.+z**2.)
        return self.M(r)/(cfg.FourPiOverThree*r**3)
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).

        Syntax:
        
            .tdyn(R,z=0.)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)     
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        return - np.pi*cfg.G*self.rho0*self.rs**2 /x * (-np.pi+\
            2.*(1.+x)*np.arctan(1./x)+2.*(1.+x)*np.log(1.+x)+\
            (1.-x)*np.log(1.+x**2))
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs   
        fac = np.pi*cfg.G*self.rho0*self.rs/x**2 * \
            (np.pi-2.*np.arctan(1./x)-2.*np.log(1.+x)-np.log(1.+x**2))/r
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] assuming isotropic velicity 
        dispersion tensor ... 
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        return self.Vmax * 0.299*np.exp(x**0.17) / (1.+0.286*x**0.797)
    def sigma_accurate(self,R,z=0.,beta=0.):
        """
        Velocity dispersion [kpc/Gyr].
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
            beta: anisotropy parameter (default=0., i.e., isotropic)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(self.dIdx_sigma, xx, np.inf,args=(beta,))[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf,args=(beta,))[0]
        sigmasqr = cfg.TwoPiG*self.rho0*(1.+x)*(1.+x**2)*self.rs**2 / \
            x**(2.*beta) * I
        return np.sqrt(sigmasqr)
    def dIdx_sigma(self,x,beta):
        """
        Integrand for the integral in the velocity dispersion of Burkert.
        """
        return (0.5*np.log(1.+x**2)+np.log(1.+x)-np.arctan(x))* \
            x**(2.*beta-2.) / ((1.+x)* (1.+x**2))

class coreNFW(object):
    """
    Class that implements the "coreNFW" profile (Read+2016):

        M(r) = M_NFW(r) g(y)
        rho(r) = rho_NFW(r) g(y) + [1-g(y)^2] M_NFW(r) / (4 pi r^2 r_c)
    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        y = r / r_c with r_c a core radius, usually smaller than r_s
        g(y) = tanh(y)
    
    Syntax:
    
        halo = coreNFW(M,c,rc,Delta=200.,z=0.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: NFW halo concentration (float)
        rc: core radius [kpc]
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rc: core radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .Vmax: maximum circular velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)      
    
    HISTORY: Arthur Fangzhou Jiang (2021-03-11, Caltech)
    """
    def __init__(self,M,c,rc,Delta=200.,z=0.):
        """
        Initialize coreNFW profile.
        
        Syntax:
        
            halo = coreNFW(M,c,rc,Delta=200.,z=0.)
        
        where
        
            M: halo mass [M_sun] (float), 
            c: halo concentration (float),  
            rc: core radius [kpc]      
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default is 200.)         
            z: redshift (float)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.rc = rc
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.xc = self.rc / self.rs
        self.rmax = self.rs * 2.163 # accurate only if r_c < r_s
        self.rho0 = self.rhoc*self.Deltah/3.*self.ch**3./self.f(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2.     
        self.Vmax = self.Vcirc(self.rmax) # accurate only if r_c < r_s
        self.s001 = self.s(0.01*self.rh)
    def f(self,x):
        """
        Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)
    
        Syntax:
    
            .f(x)
        
        where
        
            x: dimensionless radius r/r_s (float or array)
        """
        return np.log(1.+x) - x/(1.+x) 
    def g(self,y):
        """
        Auxiliary method for coreNFW profile: f(y) = tanh(y)
    
        Syntax:
    
            .g(y)
        
        where
        
            y: dimensionless radius r/r_c (float or array)
        """
        return np.tanh(y)
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        y = r / self.rc
        f = self.f(x)
        g = self.g(y)
        return self.rho0*(g/(x *(1.+x)**2)+(1.-g**2)*f/(self.xc*x**2.))
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        r1 = r*(1.+cfg.eps)
        r2 = r*(1.-cfg.eps)
        rho1 = self.rho(r1)
        rho2 = self.rho(r2)
        return - np.log(rho1/rho2) / np.log(r1/r2)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        y = r/self.rc
        return cfg.FourPi*self.rho0*self.rs**3. * self.f(x) * self.g(y)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
        
        where 
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)   
        """
        r = np.sqrt(R**2.+z**2.)
        return self.M(r)/(cfg.FourPiOverThree*r**3)
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).

        Syntax:
        
            .tdyn(R,z=0.)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)     
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi_accurate(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi_accurate(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        Phi1 = - cfg.G * self.M(r)/r
        if isinstance(x,list) or isinstance(x,np.ndarray):
            if len(x.shape)==1: # i.e., if the input R array is 1D
                I = []
                for xx in x:
                    #II = quad(self.dIdx_Phi, xx, self.ch,)[0]
                    II = quad(self.dIdx_Phi, xx, np.inf,)[0]
                    I.append(II)
                I = np.array(I)
            elif len(x.shape)==2: # i.e., if the input R array is 2D
                I = np.empty(x.shape)
                for i,xx in enumerate(x):
                    for j,xxx in enumerate(xx):
                        #II = quad(self.dIdx_Phi, xxx, self.ch,)[0]
                        II = quad(self.dIdx_Phi, xxx, np.inf,)[0]
                        I[i,j] = II
        else:
            I = quad(self.dIdx_Phi, x, self.ch,)[0]
        Phi2 = self.Phi0 * I
        return Phi1 + Phi2
    def dIdx_Phi(self,x):
        """
        Integrand for the second-term of the potential of coreNFW.
        """
        f = self.f(x)
        g = self.g(x/self.xc)
        return g/(1.+x)**2 + (1.-g**2)*f/(x*self.xc)
    def Phi(self,R,z=0.):
        """
        Approximation expression for gravitational potential 
        [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2):
        
            Phi(x) ~ [1+s(x)] Phi_core + s(x) Phi_NFW(x) 
        
        where
        
            x = r/r_s
            Phi_core ~ Phi_NFW(0.8 x_c) is the flat potential in the core
            Phi_NFW(x) is the NFW potential
        
        For exact (but slower evaluation of the) potential, use 
        .Phi_accurate 
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        xtrans = 0.8 * self.xc # an empirical transition scale
        s = 0.5 + 0.5*np.tanh((x-xtrans)/xtrans) # transition function
        Phic = self.Phi0 * np.log(1.+xtrans)/xtrans
        PhiNFW = self.Phi0 * np.log(1.+x)/x
        return (1.-s)*Phic + s*PhiNFW
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs   
        y = r / self.rc
        fac = self.Phi0 * (self.g(y)*self.f(x)/x) / r**2.
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def rmax_accurate(self):
        """
        Radius [kpc] at which maximum circular velocity is reached, which
        is given by the root of:
        
            g(y)/(1+x)^2 - f(x)g(y)/x^2 + [1-g(y)^2]f(x)/(x x_c) = 0
            
        where
        
            x = r/r_s
            x_c = r_c / r_s
            y = r/r_c = x/x_c
            g(y) = tanh(y)
        """
        xmax = brentq(self.Findxmax, 0.1, 10., args=(),
            xtol=0.001,rtol=1e-5,maxiter=1000)
        return xmax * self.rs
    def Findxmax(self,x):
        """
        The left-hand-side function for finding x_max = r_max / r_s.
        """
        f = self.f(x)
        g = self.g(x/self.xc)
        return g/(1.+x)**2-f*g/x**2+(1.-g**2)*f/x/self.xc
    def sigma_accurate(self,R,z=0.,beta=0.):
        """
        Velocity dispersion [kpc/Gyr].
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
            beta: anisotropy parameter (default=0., i.e., isotropic)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        y = r / self.rc
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(self.dIdx_sigma, xx, np.inf,args=(beta,))[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(self.dIdx_sigma, x, np.inf,args=(beta,))[0]
        f = self.f(x)
        g = self.g(y)
        A = g/(x*(1.+x)**2)+(1.-g**2)*f/(self.xc*x**2)
        sigmasqr = -self.Phi0 / x**(2.*beta) / A * I
        return np.sqrt(sigmasqr)
    def dIdx_sigma(self,x,beta):
        """
        Integrand for the integral in the velocity dispersion of Burkert.
        """
        f = self.f(x)
        g = self.g(x/self.xc)
        return (g/(x*(1.+x)**2)+(1.-g**2)*f/(self.xc*x**2))*\
            f*g*x**(2.*beta-2.)
    def sigma(self,R,z=0.):
        """
        Approximation expression for velocity dispersion [kpc/Gyr] at 
        radius r = sqrt(R^2 + z^2), assuming isotropic velicity.
        
        For exact (but slower evaluation of the) dispersion, use
        .sigma_accurate
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs
        xtrans = 0.8 * self.xc # an empirical transition scale
        s = 0.5 + 0.5*np.tanh((x-xtrans)/xtrans) # transition function
        sigmac = self.Vmax*1.4393*xtrans**0.354/(1.+1.1756*xtrans**0.725)
        sigmaNFW = self.Vmax*1.4393*x**0.354/(1.+1.1756*x**0.725)
        return (1.-s)*sigmac + s*sigmaNFW

class Dekel(object):
    """
    Class that implements Dekel+ (2016) profile:

        rho(R,z)=rho_0/[(r/r_s)^alpha * (1+(r/r_s)^(1/2))^(2(3.5-alpha))]
        M(R,z) = M_vir * g(x,alpha) / g(c,alpha) 
               = M_vir [chi(x)/chi(c)]^[2(3-alpha)] 
        
    in a cylindrical frame (R,phi,z), where   
        
        r = sqrt(R^2 + z^2)
        c: concentration parameter
        r_s: scale radius, i.e., R_vir / c, where R_vir is the virial
            radius. (Note that unlike NFW or Einasto, where r_s is r_-2,
            here r_s is not r_-2, but 2.25 r_-2 / (2-alpha)^2 ) 
        alpha: shape parameter, the innermost logarithmic density slope
        x = r/r_s
        chi(x) = x^(1/2) / (1+x^(1/2))
        g(x,alpha) = chi(x)^[2(3-alpha)]
        rho_0: normalization density, 
            rho_0 = c^3 (3-alpha) Delta rho_crit / [3 g(c,alpha)]
        M_vir: virial mass, related to rho_0 via 
            M_vir = 4 pi rho_0 r_s^3 g(c,alpha) / (3-alpha)
    
    Syntax:
    
        halo = Dekel(M,c,alpha,Delta=200.,z=0.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float)
        c: concentration (float)
        alpha: shape parameter, the inner most log density slope (float)
            (there are singularities for computing potential and 
            velocity dispersion, at 1+i/4, for i=0,1,2,...,8)
        Delta: multiples of the critical density of the Universe 
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .alphah: halo innermost logarithmic density slope
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .Vmax: maximum circular velocity [kpc/Gyr]
        .s001: logarithmic density slope at 0.01 halo radius
        .sh: the old name for s001, kept for compatibility purposes
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r =sqrt(R^2 + z^2)
        .s(R,z=0.): logarithmic density slope at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r = sqrt(R^2 + z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)  
    
    HISTORY: Arthur Fangzhou Jiang (2018-03-23, UCSC)
             Arthur Fangzhou Jiang (2019-08-26, UCSC)
    """
    def __init__(self,M,c,alpha,Delta=200.,z=0.):
        """
        Initialize Dekel+ profile.
        
        Syntax:
            
            halo = Dekel(M,c,alpha,Delta=200.,Om=0.3,h=0.7)
        
        where
        
            M: halo mass [M_sun] (float)
            c: halo concentration (float)
            alpha: innermost logarithmic density slope (float)
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default 200.)  
            z: redshift (float) (default 0.)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.alphah = alpha
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rmax = self.rs * (2.-self.alphah)**2.
        self.r2 = self.rmax/2.25
        self.rho0 = self.rhoc*self.Deltah * (3.-self.alphah)/3. * \
            self.ch**3./self.g(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2. / \
            ((3.-self.alphah)*(2.-self.alphah)*(2.*(2.-self.alphah)+1))
        self.Vmax = self.Vcirc(self.rmax)
        self.sh = (self.alphah+0.35*self.ch**0.5) / (1.+0.1*self.ch**0.5)
        self.s001 = self.s(0.01*self.rh)
    def X(self,x):
        """
        Auxiliary function for Dekel+ profile
    
            chi := x^0.5 / 1+x^0.5  
    
        Syntax:
    
            .X(x)
    
        where 
        
            x: dimensionless radius r/r_s (float or array)
        """
        u = x**0.5
        return u/(1.+u)
    def g(self,x):
        """
        Auxiliary function for Dekel+ profile
    
            g(x;alpha):= chi^[2(3-alpha)], with chi := x^0.5 / 1+x^0.5  
    
        Syntax:
    
            .g(x)
    
        where 
    
            x: dimensionless radius r/r_s (float or array)
        """
        return self.X(x)**(2.*(3.-self.alphah))
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.rho0 / ( x**self.alphah * \
            (1.+x**0.5)**(2.*(3.5-self.alphah)) )
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        sqrtx = np.sqrt(x) 
        return (self.alphah+3.5*sqrtx) / (1.+sqrtx) 
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.Mh * self.g(x)/self.g(self.ch)
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        return 3./(cfg.FourPi*r**3.) * self.M(R,z) # <<< to be replaced
            # by a simpler analytic expression, but this one is good
            # enough for now. 
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .tdyn(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.rs
        X = self.X(x)
        Vvsqr = self.Vcirc(self.rh)**2.
        u = 2*(2.-self.alphah)
        return -Vvsqr * 2*self.ch / self.g(self.ch) * \
            ((1.-X**u)/u - (1.-X**(u+1))/(u+1))
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        x = r / self.rs    
        fac = ((2.-self.alphah)*(2.*(2.-self.alphah)+1.)) * \
            self.Phi0 * (self.g(x)/x) / r**2.
        return fac*R, fac*0., fac*z 
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] at radius r = sqrt(R^2 + z^2), 
        assuming isotropic velicity dispersion tensor, following what I 
        derived based on Zhao (1996) eq.19 and eqs.A9-A11:
        
            sigma^2(r) = 2 Vv^2 c/g(c,alpha) x^3.5 / chi^(2(3.5-alpha))
                Sum_{i=0}^{i=8} (-1)^i 8! (1-chi^(4(1-alpha)+i)) / 
                ( i! (8-i)! (4(1-alpha)+i) ).
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        X = self.X(x)
        Vvsqr = self.Vcirc(self.rh)**2.
        u = 4*(1.-self.alphah)
        sigmasqr = 2.*Vvsqr*self.ch/self.g(self.ch) \
            *(x**3.5)/(X**(2.*(3.5-self.alphah))) \
            * ( (1.-X**u)/u - 8.*(1.-X**(u+1.))/(u+1.) \
            + 28.*(1.-X**(u+2.))/(u+2.) - 56.*(1.-X**(u+3.))/(u+3.) \
            + 70.*(1.-X**(u+4.))/(u+4.) - 56.*(1.-X**(u+5.))/(u+5.) \
            + 28.*(1.-X**(u+6.))/(u+6.) - 8.*(1.-X**(u+7.))/(u+7.) \
            + (1.-X**(u+8.))/(u+8.) )
        return np.sqrt(sigmasqr)  
        
class Einasto(object):
    """
    Class that implements Einasto (1969a,b) profile:

        rho(R,z) = rho_s exp{ - d(n) [ (r/r_s)^(1/n) - 1 ] }
        
    in a cylindrical frame (R,phi,z), where
        
        r = sqrt(R^2 + z^2)
        r_s: scale radius, at which d ln rho(r) / d ln(r) = -2
        n: Einasto shape index, the inverse of which, alpha=1/n, is also 
            called the Einasto shape parameter
        d(n): geometric constant which makes that r_s to be a 
            characteristic radius. (We usually use d(n)=2n, 
            which makes r_s = r_-2, i.e., the radius at which 
            d ln rho(r) / d ln(r) = -2.) 
        rho_s: density at r=r_s  (Since r_s=r_-2, rho_s is also denoted 
            as rho_-2.)
    
    See Retana-Montenegro+2012 for details.
    
    Syntax:
    
        halo = Einasto(M,c,alpha,Delta=200.,Om=0.3,h=0.7)

    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float)
        c: concentration (float)
        alpha: shape (float)
        Delta: multiples of the critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration (halo radius / scale radius)
        .alphah: halo shape
        .Deltah: spherical overdensity wrt instantaneous critical density
        .z: redshift
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: halo's average density [M_sun kpc^-3]
        .rh: halo radius [kpc], within which density is Deltah times rhoc
        .rs: scale radius [kpc], at which log density slope is -2
        .nh: inverse of shape paramter (1 / alphah)
        .hh: scale length [kpc], defined as rs / (2/alphah)^(1/alphah)
        .rho0: halo's central density [M_sun kpc^-3]
        .xmax: dimensionless rmax, defined as (rmax/hh)^alphah
        .rmax: radius [kpc] at which maximum circular velocity is reached 
        .Vmax: maximum circular velocity [kpc/Gyr]
        .Mtot: total mass [M_sun] of the Einasto profile integrated to 
            infinity
        .s001: logarithmic density slope at 0.01 halo radius
        
    Methods:

        .rho(R,z=0.): density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2)
        .M(R,z=0.): mass [M_sun] within radius r = sqrt(R^2 + z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circular velocity [kpc/Gyr] at radius r
        .sigma(R,z=0.): velocity dispersion [kpc/Gyr] at radius r=R
        .d2Phidr2(R,z=0.): second radial derivative of potential [1/Gyr^2]
            at radius r=sqrt(R^2+r^2)
    
    HISTORY: Arthur Fangzhou Jiang (2016-11-08, HUJI)
             Arthur Fangzhou Jiang (2019-09-10, HUJI)
    """
    def __init__(self,M,c,alpha,Delta=200.,z=0.):
        """
        Initialize Einasto profile.
        
        Syntax:
            
            halo = Einasto(M,c,alpha,Delta=200.,Om=0.3,h=0.7)
        
        where
        
            M: halo mass [M_sun] (float)
            c: halo concentration (float)
            alpha: Einasto shape (float)
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default 200.)  
            z: redshift (float) (default 0.)
        """
        # input attributes
        self.Mh = M 
        self.ch = c
        self.alphah = alpha
        self.Deltah = Delta
        self.z = z
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.nh = 1./self.alphah
        self.hh = self.rs / (2.*self.nh)**self.nh
        self.xh = (self.rh / self.hh)**self.alphah
        self.rho0 = self.Mh / (cfg.FourPi * self.hh**3. * self.nh * \
            gamma_lower(3.*self.nh,self.xh)) 
        self.rmax = 1.715*self.alphah**(-0.00183) * \
            (self.alphah+0.0817)**(-0.179488) * self.rs
        self.xmax = (self.rmax / self.hh)**self.alphah
        self.Mtot = cfg.FourPi * self.rho0 * self.hh**3. * self.nh \
            * gamma(3.*self.nh)
        self.GMtot = cfg.G*self.Mtot 
        self.Vmax = self.Vcirc(self.rmax)
        self.s001 = self.s(0.01*self.rh)
    def x(self,r):
        """
        Auxilary method that computes dimensionless radius 
        
            x := (r/h)^alpha 
        
        at radius r = sqrt(R^2+z^2).
            
        Syntax:
        
            .x(r)
            
        where
            
            r = sqrt(R^2 + z^2) [kpc] (float or array)
        """
        return (r / self.hh)**self.alphah 
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.)
        return self.rho0 * np.exp(-self.x(r))
    def s(self,R,z=0.):
        """
        Logarithmic density slope 
            
            - d ln rho / d ln r 
        
        at radius r = sqrt(R^2 + z^2). 
        
        Syntax:
        
            .s(R,z=0.)

        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        return self.x(r) / self.nh
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
        
            M(R,z) = M_tot gamma(3n,x)/Gamma(3n)
            
        where x = (r/h)^alpha; h = r_s/(2n)^n; and gamma(a,x)/Gamma(a) 
        together is the normalized lower incomplete gamma function, as 
        can be computed directly by scipy.special.gammainc.    
            
        Syntax:
        
            .M(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return self.Mtot * gammainc(3.*self.nh,self.x(r))
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        return 3./(cfg.FourPi*r**3.) * self.M(R,z)
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .tdyn(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at radius r = sqrt(R^2 + z^2):
        
            Phi = - G M_tot/[h Gamma(3n)] [gamma(3n,x)/x^n + Gamma(2n,x)]
        
        where x = (r/h)^alpha; h = r_s/(2n)^n; gamma(a,x)/Gamma(a) 
        together is the normalized lower incomplete gamma function; 
        Gamma(a,x) is the non-normalized upper incomplete gamma function;
        and Gamma(a) is the (complete) gamma function.
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        x = self.x(r)
        a = 3.*self.nh
        return - self.GMtot/self.hh * ( gammainc(a,x)/x**self.nh \
            + gamma_upper(2*self.nh,x)/gamma(a) )
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        fac = - self.GMtot * gammainc(3.*self.nh,self.x(r)) / r**3.
        return fac*R, fac*0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(self.GMtot/r *gammainc(3.*self.nh,self.x(r)))
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] assuming isotropic velicity 
        dispersion tensor ... 
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        x = self.x(r)
        if isinstance(x,list) or isinstance(x,np.ndarray):
            I = []
            for xx in x:
                II = quad(dIdx_Einasto, xx, np.inf, args=(self.nh,),)[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(dIdx_Einasto, x, np.inf, args=(self.nh,),)[0]
        sigmasqr = self.GMtot/self.hh*self.nh*np.exp(x) * I 
        return np.sqrt(sigmasqr)
    def d2Phidr2(self,R,z=0):
        """
        Second radial derivative of the gravitational potential [1/Gyr^2]
        computed at (R,z).
            
        Syntax:
        
            .d2Phidr2(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r = np.sqrt(R**2.+z**2.)
        x = self.x(r)
        fact1 = self.GMtot/(r**3 * gamma(3*self.nh))
        fact2 = x**(3*self.nh)*np.exp(-x)/self.nh
        fact3 = 2.*gamma_lower(3*self.nh, x)
        return fact1*(fact2 - fact3)

def dIdx_Einasto(x,n):
    """
    Integrand for the integral in the velocity dispersion of Einasto.
    """
    return gammainc(3.*n,x)/(np.exp(x)*x**(n+1.))
        
class MN(object):
    """
    Class that implements Miyamoto & Nagai (1975) disk profile:

        Phi(R,z) = - G M / sqrt{ R^2 + [ a + sqrt(z^2+b^2) ]^2 }
    
    in a cylindrical frame (R,phi,z), where 
    
        M: disk mass
        a: scalelength 
        b: scaleheight.
    
    Syntax:
    
        disk = MN(M,a,b)

    where
    
        M: disk mass [M_sun] (float)
        a: scalelength [kpc] (float)
        b: scaleheight [kpc] (float)

    Attributes:
    
        .Md: disk mass [M_sun]
        .Mh: the same as .Md, but for the purpose of keeping the notation
            for "host" mass consistent with the other profile classes
        .a: disk scalelength [kpc]
        .b: disk scaleheight [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at (R,z) 
        .M(R,z=0.): mass [M_sun] within radius r=sqrt(R^2+z^2),
            defined as M(r) = r Vcirc(r,z=0)^2 / G
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at (R,z)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at (R,z=0), defined as
            sqrt(R d Phi(R,z=0.)/ d R)
        .sigma(R,z=0.): velocity dispersion [kpc/Gyr] at (R,z) 
        .d2Phidr2(R,z=0.): second radial derivative of potential [1/Gyr^2]
            at (R,z)
    
    HISTORY: Arthur Fangzhou Jiang (2016-11-03, HUJI)
             Arthur Fangzhou Jiang (2019-08-27, UCSC)
    """
    def __init__(self,M,a,b):
        """
        Initialize Miyamoto-Nagai disk profile
        
        Syntax:
        
            disk = MN(M,a,b)
        
        where 
        
            M: disk mass [M_sun], 
            a: disk scalelength [kpc]
            b: disk scaleheight [kpc]
        """
        # input attributes
        self.Md = M
        self.Mh = self.Md 
        self.a = a
        self.b = b
        #
        # supportive attributes repeatedly used by following methods
        self.GMd = cfg.G * self.Md

        # we build the mass profile interpolation lazily
        self.Minterp = None

    def s1sqr(self,z):
        """
        Auxilary method that computes (a + sqrt(z^2+b^2))^2 at height z.
        
        Syntax:
        
            .s1sqr(z) 
        
        where
        
            z: z-coordinate [kpc] (float or array)
        """
        return (self.a + self.s2(z))**2.
    def s2(self,z):
        """
        Auxilary method that computes zeta = sqrt(z^2+b^2) at height z.
            
        Syntax:
        
            .s2(z)
        
        where 
        
            z: z-coordinate [kpc] (float or array)
        """
        return np.sqrt(z**2. + self.b**2)             
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at (R,z).
            
        Syntax:
        
            .rho(R,z)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        Rsqr = R**2.
        s1sqr = self.s1sqr(z)
        s2 = self.s2(z)
        return self.Md * self.b**2. * (self.a*Rsqr+(self.a+3.*s2)*s1sqr)\
            / (cfg.FourPi * (Rsqr+s1sqr)**2.5 * s2**3.) 
    def M(self,R,z=0.):
        """
        Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0):   
        
        where
                
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)  
        """
        if not self.Minterp:
            # lazy-build interpolator for spherically enclosed mass,
            # which is not analytical calculable
            # define the function, integrate it, interpolate
            def integrand_1d(z, r):
                q = np.sqrt(r**2 - z**2)
                x = np.sqrt(z**2 + self.b**2)
                top = self.a**3 + self.a*q**2 + 3.*x*self.a**2 + 3.*self.a*x**2 +\
                        x**3 - (q**2 + (self.a + x)**2)**(1.5)
                bottom = x**3 * (q**2 + (self.a + x)**2)**1.5
                return top / bottom
            interp_rads = self.a * np.logspace(-3, 3.5, 100)
            interp_mass = np.zeros(len(interp_rads))
            for i in range(len(interp_rads)):
                r = interp_rads[i]
                interp_mass[i] = quad(lambda z: integrand_1d(z, r), 0, r)[0]
            interp_mass *= (-1. * self.b**2 * self.Md)
            self.Minterp = InterpolatedUnivariateSpline(np.log10(interp_rads),
                                                        np.log10(interp_mass))

        r = np.sqrt(R**2.+z**2.) 
        return 10.**self.Minterp(np.log10(r))
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        return 3./(cfg.FourPi*r**3.) * self.M(R,z) 
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .tdyn(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z)) 
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at (R,z).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        Rsqr = R**2.
        s1sqr= self.s1sqr(z)
        return -self.GMd / np.sqrt(Rsqr+s1sqr)
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        Rsqr = R**2.
        s1sqr= self.s1sqr(z)   
        s1 = np.sqrt(s1sqr)
        s2 = self.s2(z)
        fac = -self.GMd / (Rsqr+s1sqr)**1.5
        return fac*R, fac*0., fac*z * s1/s2
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at (R,z=0.), defined as 
            
            V_circ(R,z=0.) = sqrt(R d Phi(R,z=0.)/ d R)
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.)
                
        Note that only z=0 is meaningful. Because circular velocity is 
        the speed of a satellite on a circular orbit, and for a disk 
        potential, a circular orbit is only possible at z=0. 
        """
        return np.sqrt(R*-self.fgrav(R,z)[0])
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] at (R,z), following 
        Ciotti & Pellegrini 1996 (CP96).
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.) 
        
        Note that this is at the same time the R-direction and the 
        z-direction velocity dispersion, as we implicitly assumed
        that the distribution function of the disk potential depends only
        on the isolating integrals E and L_z. If we further assume 
        isotropy, then it is also the phi-direction velocity dispersion.
        (See CP96 eqs 11-17 for more.)
        """
        Rsqr = R**2.
        s1sqr = self.s1sqr(z)
        s2 = self.s2(z)
        sigmasqr = cfg.G*self.Md**2 *self.b**2 /(8.*np.pi*self.rho(R,z))\
            * s1sqr / ( s2**2. * (Rsqr + s1sqr)**3.)
        return np.sqrt(sigmasqr)
    def Vphi(self,R,z=0):
        """
        The mean azimuthal velocity [kpc/Gyr] at (R,z), following 
        Ciotti & Pellegrini 1996 eq.17.
        
        Syntax: 
        
            .Vphi(R,z=0)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.) 
        
        Note that this is different from the circular velocity by an 
        amount of asymmetric drift, i.e.,
        
            V_a = V_circ - V_phi.
            
        Note that we have made the assumption of isotropy. 
        """
        Rsqr = R**2.
        s1sqr = self.s1sqr(z)
        s2 = self.s2(z)
        Vphisqr = cfg.G*self.Md**2 *self.a *self.b**2 / \
            (cfg.FourPi*self.rho(R,z)) * Rsqr/(s2**3. *(Rsqr+s1sqr)**3.)
        return np.sqrt(Vphisqr)
    def d2Phidr2(self,R,z=0):
        """
        Second radial derivative of the gravitational potential [1/Gyr^2]
        computed at (R,z).
            
        Syntax:
        
            .d2Phidr2(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        s1sqr = self.s1sqr(z)
        s1    = np.sqrt(s1sqr)
        s2    = self.s2(z)
        s2sqr = s2**2.
        R2    = R**2.
        z2    = z**2.
        r     = np.sqrt(R2 + z2)

        d2PhidR2  = (1./(R2 + s1sqr)**1.5) - (3.*R2/(R2 + s1sqr)**2.5)
        d2PhidRdz =  -3.*R*z*s1/s2/(R2+s1sqr)**2.5
        d2Phidz2  = (s1/s2/(R2+s1sqr)**1.5)+(z2/s2sqr/(R2+s1sqr)**1.5) - \
                    (3.*z2*s1sqr/s2sqr/(R2+s1sqr)**2.5) - \
                    (z2*s1/s2**3./(R2+s1sqr)**1.5)

        # Jacobian entries found by considering R=rcos(theta), z=rsin(theta)
        dRdr = R/r
        dzdr = z/r

        return cfg.G*self.Md*(d2PhidR2*dRdr**2. + 2.*d2PhidRdz*dRdr*dzdr + \
                              d2Phidz2*dzdr**2.)

class Hernquist(object):
    """
    Class that implements the Hernquist (1990) profile:

        rho(r) = M / (2 pi a^3) / [x (1+x)^3], x = r/a  
    
    in a cylindrical frame (R,phi,z), where 
    
        M: total mass
        a: scale radius
    
    Syntax:
    
        baryon = Hernquist(M,a)

    where
    
        M: baryon mass [M_sun] (float)
        a: scalelength [kpc] (float)

    Attributes:
    
        .Mb: baryon mass [M_sun]
        .Mh: the same as .Md, but for the purpose of keeping the notation
            for "host" mass consistent with the other profile classes
        .a: scalelength [kpc]
        .r0: the same as .a
        .rho0: characteristic density, M/(2 pi a^3) [M_sun/kpc^3]
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at (R,z) 
        .M(R,z=0.): mass [M_sun] within radius r=sqrt(R^2+z^2),
            defined as M(r) = r Vcirc(r,z=0)^2 / G
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2 + z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at (R,z)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at (R,z=0), defined as
            sqrt(R d Phi(R,z=0.)/ d R)
        .sigma(R,z=0.): velocity dispersion [kpc/Gyr] at (R,z) 
    
    HISTORY: Arthur Fangzhou Jiang (2020-09-09, Caltech)
    """
    def __init__(self,M,a):
        """
        Initialize Hernquist profile
        
        Syntax:
        
            baryon = Hernquist(M,a)
        
        where 
        
            M: baryon mass [M_sun], 
            a: scale radius [kpc]
        """
        # input attributes
        self.Mb = M
        self.Mh = self.Mb 
        self.a = a
        self.r0 = a
        # 
        # derived attributes
        self.rho0 = M/(cfg.TwoPi*a**3)
        self.rhalf = 2.414213562373095 * a      
        #
        # supportive attributes repeatedly used by following methods
        self.GMb = cfg.G * M 
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at (R,z).
            
        Syntax:
        
            .rho(R,z)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.a
        return self.rho0 / (x * (1.+x)**3)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0):   
        
        where
                
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)  
        """
        r = np.sqrt(R**2.+z**2.) 
        return self.Mb * r**2 / (r+self.a)**2
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        r = np.sqrt(R**2.+z**2.)
        return 3./(cfg.FourPi*r**3.) * self.M(R,z) 
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .tdyn(R,z=0.)
            
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)      
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z)) 
    def Phi(self,R,z=0.):
        """
        Potential [(kpc/Gyr)^2] at (R,z).
            
        Syntax:
        
            .Phi(R,z=0.)  

        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return -self.GMb / (r+self.a)
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        pass
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at (R,z=0.), defined as 
            
            V_circ(R,z=0.) = sqrt(R d Phi(R,z=0.)/ d R)
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.)
                
        Note that only z=0 is meaningful. Because circular velocity is 
        the speed of a satellite on a circular orbit, and for a disk 
        potential, a circular orbit is only possible at z=0. 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(cfg.G*self.M(r)/r)
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] assuming isotropic velicity 
        dispersion tensor ... 
        
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0.) 
        
        Note that this is at the same time the R-direction and the 
        z-direction velocity dispersion, as we implicitly assumed
        that the distribution function of the disk potential depends only
        on the isolating integrals E and L_z. If we further assume 
        isotropy, then it is also the phi-direction velocity dispersion.
        (See CP96 eqs 11-17 for more.)
        """
        r = np.sqrt(R**2.+z**2.)
        x = r/self.a
        sigmasqr = cfg.GMb/(12.*self.a)*(12.*x*(1.+x)**3*np.log(1.+1./x)\
            -(x/(1.+x))*(25.+52.*x+42.*x**2+12.*x**3)) 
        return np.sqrt(sigmasqr)

class exp(object):
    """
    Class that implements the exponential disk profile:

        Sigma(r) = Sigma_0 exp(-x), x = r/a  
    
    in a cylindrical frame (R,phi,z), where 
    
        M: total mass
        a: scale radius
    
    Syntax:
    
        disk = exp(M,a)

    where
    
        M: baryon mass [M_sun] (float)
        a: scale radius [kpc] (float)

    Attributes:
    
        .Md: baryon mass [M_sun]
        .Mb: the same as .Md
        .Mh: the same as .Md, but for the purpose of keeping the notation
            for "host" mass consistent with the other profile classes
        .a: scalelength [kpc]
        .Rd: the same as .a
        .Sigma0: central surface density, M/(2 pi a^2) [M_sun/kpc^2]
        .rhalf: half-mass radius [kpc]
        
    Methods:
    
        .Sigma(R,z=0.): surface density [M_sun kpc^-2] at (R,z) 
        .M(R,z=0.): mass [M_sun] within radius r=sqrt(R^2+z^2),
            defined as M(r) = r Vcirc(r,z=0)^2 / G
    
    Note: incomplete, other methods and attributes to be added...
    
    HISTORY: Arthur Fangzhou Jiang (2020-09-09, Caltech)
    """
    def __init__(self,M,a):
        """
        Initialize Hernquist profile
        
        Syntax:
        
            baryon = Hernquist(M,a)
        
        where 
        
            M: baryon mass [M_sun], 
            a: scale radius [kpc]
        """
        # input attributes
        self.Md = M
        self.Mb = M
        self.Mh = M
        self.a = a
        self.r0 = a
        # 
        # derived attributes
        self.Sigma0 = M/(cfg.TwoPi*a**2)
        self.rhalf = 1.678 * a      
        #
        # supportive attributes repeatedly used by following methods
        self.GMd = cfg.G * M 
    def Sigma(self,R):
        """
        Surface density [M_sun kpc^-2] at R.
            
        Syntax:
        
            .Sigma(R,z)
        
        where
        
            R: R-coordinate [kpc] (float or array)
        """
        x = R/self.a
        return M * exp(-x)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0):   
        
        where
                
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)  
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r/self.a
        return self.Md * (1.-(1.+x)*np.exp(-x))


class Green(object):
    """
    Class that implements the Green and van den Bosch (2019) profile,
    which incorporates tidal evolution on top of a standard Navarro, 
    Frenk, & White (1997) profile:

        rho(R,z) = H(r | f_b, c_s) * rho_{NFW}(R,z)

        where

        rho_{NFW}(R,z) = rho_crit * delta_char / [(r/r_s) * (1+r/r_s)^2]
                       = rho_0 / [(r/r_s) * (1+r/r_s)^2]

        and

        H(r | f_b, c) = f_{te} / [1+( r * [(r_{vir} - r_{te})/
                          (r_{vir} * r_{te})])^delta]

    
    in a cylindrical frame (R,phi,z), where 
    
        r = sqrt(R^2 + z^2)
        r_s: scale radius, at which d ln rho(r) / d ln(r) = -2
        rho_crit: critical density of the Universe
        delta_char = Delta_halo / 3 * c^3 / f(c), where c = R_vir / r_s 
            is the concentration parameter
        H: transfer function that converts from NFW to stripped profile
        f_b: bound mass fraction relative to peak mass at infall
        f_{te}, r_{te}, delta: free parameters calibrated against DASH
            simulations, all are functions of f_b and c
    
    Syntax:
    
        halo = Green(Mi,c,Delta=200.,z=0.)
        
    where 
    
        Mi: initial halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: halo concentration (float)
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: CURRENT halo mass [M_sun]
        .Minit: INITIAL halo mass [M_sun]
        .ch: INITIAL halo concentration (undefined once halo begins to be 
             stripped)
        The remaining below are properties of the initial NFW halo prior
        to the onset of stripping.
        .Deltah: spherical overdensity wrt instantaneous critical density
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .sigma0: physical units for velocity dispersion for DASH conversion [kpc/Gyr]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        #.Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        # Phi is not implemented currently, since not needed.
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z)
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .d2Phidr2(R,z=0.): second radial derivative of potential [1/Gyr^2]
            at radius r=sqrt(R^2+r^2)
    
    HISTORY: Sheridan Beckwith Green (2020-04, Yale)
    """
    def __init__(self,Mi,c,Delta=200.,z=0.):
        """
        Initialize Green profile.
        
        Syntax:
        
            halo = Green(Mi,c,Delta=200.,z=0.)
        
        where
        
            Mi: INITIAL halo mass [M_sun] (float), 
            c: INITIAL halo concentration at infall(float),        
            Delta: spherical overdensity with respect to the critical 
                density of the universe (default is 200.)         
            z: redshift of infall (float)
        """
        # input attributes
        self.Minit = Mi
        self.Mh = Mi
        self.fb = 1.
        self.log10fb = np.log10(self.fb)
        self.ch = c
        self.log10ch = np.log10(self.ch)
        self.Deltah = Delta
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Minit / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rmax = self.rs * 2.163
        self.sigma0 = np.sqrt(cfg.G * self.Minit / self.rh)
        #
        # attributes repeatedly used by following methods
        self.rho0 = self.rhoc*self.Deltah/3.*self.ch**3./self.f(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2.

    def transfer(self, x):
        """
        Transfer function from Green and van den Bosch (2019), defined
        by equations (5-8) and Table 1. This is used to compute the
        stripped density profile

        Syntax:

            .transfer(x)

        where

            x: dimensionless radius r/r_s (float or array)
        """

        fte = 10**(cfg.gvdb_fp[0] * (self.ch / 10.)**cfg.gvdb_fp[1] * self.log10fb + cfg.gvdb_fp[2] * (1. - self.fb)**cfg.gvdb_fp[3] * self.log10ch)
        rte = 10**(self.log10ch + cfg.gvdb_fp[4] * (self.ch / 10.)**cfg.gvdb_fp[5] * self.log10fb + cfg.gvdb_fp[6] * (1. - self.fb)**cfg.gvdb_fp[7] * self.log10ch) * np.exp(cfg.gvdb_fp[8] * (self.ch / 10.)**cfg.gvdb_fp[9] * (1. - self.fb))
        delta = 10**(cfg.gvdb_fp[10] + cfg.gvdb_fp[11]*(self.ch / 10.)**cfg.gvdb_fp[12] * self.log10fb + cfg.gvdb_fp[13] * (1. - self.fb)**cfg.gvdb_fp[14] * self.log10ch)

        rte = min(rte, self.ch)

        return fte / (1. + (x * ((self.ch - rte)/(self.ch*rte)))**delta)

    def rte(self):
        """
        Returns just the r_{te} quantity from the transfer function 
        of Green and van den Bosch (2019), defined by equation (7)
        and Table 1. The r_{te} will be in physical units.

        Syntax:

            .rte()

        """

        rte = 10**(cfg.gvdb_fp[4] * (self.ch / 10.)**cfg.gvdb_fp[5] * self.log10fb + cfg.gvdb_fp[6] * (1. - self.fb)**cfg.gvdb_fp[7] * self.log10ch) * np.exp(cfg.gvdb_fp[8] * (self.ch / 10.)**cfg.gvdb_fp[9] * (1. - self.fb))

        rte = min(rte, 1.)

        return rte*self.rh

    def update_mass(self, Mnew):
        """
        Updates Green profile Mh to be the new mass after some
        stripping has occurred. The bound fraction is updated according
        to this new Mh value, and the log10(f_b) is updated as well in
        order to save computation time when computing densities.

        NOTE:
            An alternative implementation would be to set subhalo masses
            to zero once their m/m_{acc} falls below phi_{res}.
            Using the current implementation, during analysis, one must
            mask all subhaloes with m <= M_{init} * phi_{res}, since these
            are subhaloes that have fallen below our resolution limit.
        """
        self.Mh = Mnew
        self.fb = self.Mh / self.Minit
        if(self.fb < cfg.phi_res):
            self.fb = cfg.phi_res
            self.Mh = self.Minit * self.fb 

        self.log10fb = np.log10(self.fb)
        return self.Mh

    def f(self,x):
        """
        Auxiliary method for NFW profile: f(x) = ln(1+x) - x/(1+x)
    
        Syntax:
    
            .f(x)
        
        where
        
            x: dimensionless radius r/r_s (float or array)
        """
        return np.log(1.+x) - x/(1.+x) 
    def rho(self,R,z=0.):
        """
        Density [M_sun kpc^-3] at radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rho(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)
        """
        r = np.sqrt(R**2.+z**2.) 
        x = r / self.rs
        return self.transfer(x) * self.rho0 / (x * (1.+x)**2.)
    def M(self,R,z=0.):
        """
        Mass [M_sun] within radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .M(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r_by_rvir = np.sqrt(R**2.+z**2.) / self.rh
        if(isinstance(r_by_rvir, float)):
            return self._from_interp(r_by_rvir, 'mass')
        else:
            # assume array
            enc_masses = np.zeros(len(r_by_rvir))
            for i in range(0,len(r_by_rvir)):
                enc_masses[i] = self._from_interp(r_by_rvir[i], 'mass')
            return enc_masses
    def _from_interp(self, r_by_rvir, type='mass'):
        """
        Computes the enclosed mass or isotropic velocity dispersion at
        r/r_{vir} using interpolations of the mass/dispersion profile 
        computed from the Green and van den Bosch (2019) density model.
            
        Syntax:
        
            ._from_interp(r_by_rvir,type)
        
        where
        
            r_by_rvir: spherical radius normalized by virial radius (float)
            type: 'mass', 'sigma', 'd2Phidr2', denoting which profile to compute
        """

        if(type == 'mass'):
            interp = cfg.fb_cs_interps_mass
            phys_unit_mult = self.Minit
        elif(type == 'sigma'):
            interp = cfg.fb_cs_interps_sigma
            phys_unit_mult = self.sigma0
        elif(type == 'd2Phidr2'):
            interp = cfg.fb_cs_interps_d2Phidr2
            phys_unit_mult = cfg.G * self.Minit / self.rh**3.
        else:
            sys.exit("Invalid interpolation type specified!")

        if(r_by_rvir < cfg.rv_min):
            warnings.warn("A radius value r/rvir=%.2e is smaller than the interpolator bound in %s!" % (r_by_rvir, type))
            r_by_rvir = cfg.rv_min
        elif(r_by_rvir > cfg.rv_max):
            warnings.warn("A radius value r/rvir=%.2e is larger than the interpolator bound in %s!" % (r_by_rvir, type))
            r_by_rvir = cfg.rv_max
        
        # determine which slices in r-space we lie between
        ind_high = np.searchsorted(cfg.r_vals_int, r_by_rvir)
        ind_low = ind_high - 1

        # compute mass given f_b, c on each of the two planes in r
        val1 = interp[ind_low](self.log10fb, self.log10ch)
        val2 = interp[ind_high](self.log10fb, self.log10ch)

        # linearly interpolate between the two planes
        val = val1 + (val2 - val1) * (r_by_rvir - cfg.r_vals_int[ind_low]) / (cfg.r_vals_int[ind_high] - cfg.r_vals_int[ind_low])

        return val[0][0] * phys_unit_mult
    def rhobar(self,R,z=0.):
        """
        Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2). 
            
        Syntax:
        
            .rhobar(R,z=0.)
        
        where 
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)   
        """
        r = np.sqrt(R**2.+z**2.)
        return 3.*self.M(r) / (cfg.FourPi * r**3)
    def tdyn(self,R,z=0.):
        """
        Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).

        Syntax:
        
            .tdyn(R,z=0.)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)     
        """
        return np.sqrt(cfg.ThreePiOverSixteenG / self.rhobar(R,z))
    def fgrav(self,R,z):
        """
        gravitational acceleration [(kpc/Gyr)^2 kpc^-1] at location (R,z)
        
            [- d Phi(R,z) / d R, 0, - d Phi(R,z) / d z]
        
        Syntax:
            
            .fgrav(R,z)
            
        where
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
        
        Note that unlike the other methods, where z is optional with a 
        default of 0, here z must be specified.
        
        Return:
        
            R-component of gravitational acceleration
            phi-component of gravitational acceleration
            z-component of gravitational acceleration
        """
        r = np.sqrt(R**2.+z**2.)
        fac = -cfg.G * self.M(r) / r**3.
        return fac*R, 0., fac*z
    def Vcirc(self,R,z=0.):
        """
        Circular velocity [kpc/Gyr] at radius r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .Vcirc(R,z=0.)
            
        where

            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        return np.sqrt(r*-self.fgrav(r,0.)[0])
    def sigma(self,R,z=0.):
        """
        Velocity dispersion [kpc/Gyr] at radius r = sqrt(R^2 + z^2), 
        assuming isotropic velicity dispersion tensor, computed from
        an interpolation of the velocity dispersion calculated using
        equation (B6) of vdBosch+2018 from the Green and van den Bosch
        (2019) profile.
                
        Syntax:
            
            .sigma(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r) 
        """
        r = np.sqrt(R**2.+z**2.)
        r_by_rvir = r / self.rh
        
        return self._from_interp(r_by_rvir, 'sigma')
    def d2Phidr2(self,R,z=0.):
        """
        Second radial derivative of the gravitational potential [1/Gyr^2]
        computed at r = sqrt(R^2 + z^2).
            
        Syntax:
        
            .d2Phidr2(R,z=0.)
        
        where
        
            R: R-coordinate [kpc] (float or array)
            z: z-coordinate [kpc] (float or array)
                (default=0., i.e., if z is not specified otherwise, the 
                first argument R is also the halo-centric radius r)       
        """
        r_by_rvir = np.sqrt(R**2.+z**2.) / self.rh
        if(isinstance(r_by_rvir, float)):
            return self._from_interp(r_by_rvir, 'd2Phidr2')
        else:
            # assume array
            d2Phidr2_vals = np.zeros(len(r_by_rvir))
            for i in range(0,len(r_by_rvir)):
                d2Phidr2_vals[i] = self._from_interp(r_by_rvir[i], 'd2Phidr2')
            return d2Phidr2

#--- functions dealing with composite potential (i.e., potential list)---

def rho(potential,R,z=0.):
    """
    Density [M_sun/kpc^3], at location (R,z) in an axisymmetric potential
    which consists of either a single component or multiple components.
    
    Syntax:

        rho(potential,R,z=0.)
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the density at 
    (R,z) in this combined halo+disk host, we use: 
    
        rho([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.rho(R,z)
    return sum

def s(potential,R,z=0.):
    """
    Logarithmic density slope 
            
        - d ln rho / d ln r 
        
    at radius r = sqrt(R^2 + z^2). 
        
    Syntax:
        
        s(potential,R,z=0.)

    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the density at 
    (R,z) in this combined halo+disk host, we use: 
    
        s([halo,disk],R,z)
    """
    r = np.sqrt(R**2.+z**2.)
    r1 = r * (1.+cfg.eps)
    r2 = r * (1.-cfg.eps)
    return -np.log(rho(potential,r1)/rho(potential,r2)) / np.log(r1/r2)

def M(potential,R,z=0.):
    """
    Mass [M_sun] within spherical radius r = sqrt(R^2 + z^2) in an 
    axisymmetric potential which consists of either a single component or 
    multiple components.
            
    Syntax:
        
        M(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the mass within  
    r = sqrt(R^2 + z^2) in this combined halo+disk host, we use: 
    
        M([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.M(R,z)
    return sum
    
def rhobar(potential,R,z=0.):
    """
    Average density [M_sun kpc^-3] within radius r = sqrt(R^2 + z^2) in 
    an axisymmetric potential which consists of either a single component 
    or multiple components.
    
    Syntax:
    
        rhobar(potential,R,z=0.)
        
    where 
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
            
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the mean density 
    within r = sqrt(R^2 + z^2) in this combined halo+disk host, we use: 
    
        rhobar([halo,disk],R,z) 
    """
    r = np.sqrt(R**2.+z**2.)
    return 3./(cfg.FourPi*r**3.) * M(potential,R,z)

def tdyn(potential,R,z=0.):
    """
    Dynamical time [Gyr] within radius r = sqrt(R^2 + z^2).
    
    Syntax:
        
        tdyn(potential, R, z=0.)
            
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r) 
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the dynamical time 
    within r = sqrt(R^2 + z^2) in this combined halo+disk host, we use: 
    
        tdyn([halo,disk],R,z) 
    """
    return np.sqrt(cfg.ThreePiOverSixteenG / rhobar(potential,R,z))

def Phi(potential,R,z=0.):
    """
    Potential [(kpc/Gyr)^2] at (R,z) in an axisymmetric potential
    which consists of either a single component or multiple components.
            
    Syntax:
        
        Phi(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the gravitational
    potential at (R,z) in this combined halo+disk host, we use: 
    
        Phi([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.Phi(R,z)
    return sum

def d2Phidr2(potential,R,z=0.):
    """
    Second radial derivative of the gravitational potential [1/Gyr^2]
    at (R,z) in an axisymmetric potential which consists of either a
    single component or multiple components.
            
    Syntax:
        
        d2Phidr2(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to compute d^2(Phi)/dr^2
    at (R,z) in this combined halo+disk host, we use: 
    
        d2Phidr2([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.d2Phidr2(R,z)
    return sum
    
def Vcirc(potential,R,z=0.):
    """
    Circular velocity [kpc/Gyr] at (R,z=0), defined as 
            
        V_circ(R,z=0) = sqrt(R d Phi(R,z=0)/ d R)
    
    in an axisymmetric potential which consists of either a single 
    component or multiple components.
            
    Syntax:
        
        Vcirc(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the circular 
    velocity at (R,z) in this combined halo+disk host, we use: 
    
        Vcirc([halo,disk],R,z)
    """
    R1 = R*(1.+cfg.eps)
    R2 = R*(1.-cfg.eps)
    Phi1 = Phi(potential,R1,z)
    Phi2 = Phi(potential,R2,z)
    dPhidR = (Phi1-Phi2) / (R1-R2)
    return np.sqrt(R * dPhidR)

def sigma(potential,R,z=0.):
    """
    1D velocity dispersion [kpc/Gyr] at (R,z=0), in an axisymmetric 
    potential which consists of either a single component or multiple 
    components. For composite potential, the velocity dispersion is the
    quadratic sum of that of individual components
    
        sigma^2 = Sum sigma_i^2
            
    Syntax:
        
        sigma(potential,R,z=0):   
        
    where
        
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        R: R-coordinate [kpc] (float or array)
        z: z-coordinate [kpc] (float or array)
            (default=0., i.e., if z is not specified otherwise, the 
            first argument R is also the halo-centric radius r)  
    
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the circular 
    velocity at (R,z) in this combined halo+disk host, we use: 
    
        sigma([halo,disk],R,z)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    sum = 0.
    for p in potential:
        sum += p.sigma(R,z)**2
    return np.sqrt(sum)
        
def fDF(potential,xv,m):
    """
    Dynamical-friction (DF) acceleration [(kpc/Gyr)^2 kpc^-1] given 
    satellite mass, phase-space coordinate, and axisymmetric host
    potential:
    
        f_DF = -4piG^2 m Sum_i rho_i(R,z)F(<|V_i|)ln(Lambda_i)V_i/|V_i|^3  
    
    where
        
        V_i: relative velocity (vector) of the satellite with respect to 
            the host component i
        F(<|V_i|) = erf(X) - 2X/sqrt{pi} exp(-X^2) with 
            X = |V_i| / (sqrt{2} sigma(R,z))
        ln(Lambda_i): Coulomb log of host component i 
        
    Syntax:
    
        fDF(potential,xv,m)
          
    where 
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        m: satellite mass [M_sun] (float)
            
    Return: 
        
        R-component of DF acceleration (float), 
        phi-component of DF acceleration (float), 
        z-component of DF acceleration (float)

    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the DF acceleration
    experienced by a satellite of mass m at xv in this combined 
    halo+disk host, we do: 
    
        fDF([halo,disk],xv,m)
    
    Note: for a composite potential, we compute the DF acceleration 
    exerted by each component separately, and sum them up as the 
    total DF acceleration. This implicitly assumes that 
        1. each component has a Maxwellian velocity distribution,
        2. the velocity dispersion of each component is not affected by
           other components 
        3. the Coulomb log of each component can be treated individually.
    All these assumptions are not warranted, but there is no trivial, 
    better way to go, see e.g., Penarrubia+2010 (MN,406,1290) Appendix A.
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    #
    R, phi, z, VR, Vphi, Vz = xv
    #
    fac = -cfg.FourPiGsqr * m # common factor in f_DF 
    sR = 0. # sum of R-component of DF accelerations 
    sphi = 0. # ... phi- ...
    sz = 0. # ... z- ...
    for p in potential:
        if isinstance(p,(MN,)): # i.e., if component p is a disk
            lnL = 0.5
            #lnL = 0.0 # <<< test: turn off disk's DF 
            VrelR = VR
            Vrelphi = Vphi - p.Vphi(R,z)
            Vrelz = Vz
        else: # i.e., if component p is a spherical component
            if(cfg.lnL_type == 0):
                lnL = np.log(p.Mh/m)
            elif(cfg.lnL_type == 1):
                lnL = np.log(1. + p.Mh/m)
            elif(cfg.lnL_type == 2):
                lnL = 2.
            elif(cfg.lnL_type == 3):
                lnL = 2.5
            elif(cfg.lnL_type == 4):
                lnL = 3.
            elif(cfg.lnL_type == 5):
                lnL = np.log(p.M(R, z) / m)
            VrelR = VR
            Vrelphi = Vphi
            Vrelz = Vz
        Vrel = np.sqrt(VrelR**2.+Vrelphi**2.+Vrelz**2.)
        Vrel = max(Vrel,cfg.eps) # safety
        X = Vrel / (cfg.Root2 * p.sigma(R,z))
        fac_s = p.rho(R,z) * (cfg.lnL_pref * lnL) * ( erf(X) - \
            cfg.TwoOverRootPi*X*np.exp(-X**2.) ) / Vrel**3 
        #fac_s = 2.*fac_s # <<< test: enhance DF by a factor of 2
        #fac_s = 0.5*fac_s # <<< test: weaken DF by half
        sR += fac_s * VrelR 
        sphi += fac_s * Vrelphi 
        sz += fac_s * Vrelz 
    return fac*sR, fac*sphi, fac*sz

def fRP(potential,xv,sigmamx,Xd=1.):
    """
    Ram-pressure (RP) acceleration [(kpc/Gyr)^2 kpc^-1] of a satellite 
    due to dark-matter self-interaction (Kummer+18 eq.18), in an 
    axisymmetric host potential:
    
        f_RP = - X_d (sigma/m_x) rho_i(R,z) |V_i| V_i
    
    where
        
        V_i: relative velocity (vector) of the satellite with respect to 
            the host component i;
        X_d(V,v_esc): an order-unity factor depending on subhalo's 
            orbital velocity and own escape velocity; 
        sigma/m_x: cross section over dark-matter particle mass.
        
    Syntax:
    
        fRP(potential,xv,sigmamx,Xd=1.)
          
    where 
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        sigmamx: self-interaction cross section over particle mass 
            [cm^2/g] or [2.0884262122368293e-10 kpc^2/M_sun] (default=1.)
        Xd: deceleration fraction (default=1.)
            
    Return: 
        
        R-component of RP acceleration (float), 
        phi-component of RP acceleration (float), 
        z-component of RP acceleration (float)

    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the RP acceleration
    experienced by a satellite of self-interaction cross section of 
    sigmamx, at xv, in this combined halo+disk host, we do: 
    
        fRP([halo,disk],xv,sigmamx,Xd=1.)
    
    Note: for a composite potential, we compute the RP acceleration 
    exerted by each dark-matter component separately, and sum them up as 
    the total RP acceleration. We skip any baryon component, such as MN
    disk, unless the MN disk is a dark-matter disk. 
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    #
    R, phi, z, VR, Vphi, Vz = xv
    # 
    fac = -Xd * (sigmamx*2.0884262122368293e-10) # common factor in f_RP
    sR = 0. # sum of R-component of RP accelerations 
    sphi = 0. # ... phi- ...
    sz = 0. # ... z- ...
    for p in potential:
        if isinstance(p,(MN,)): # i.e., if component p is a baryon disk
            # then there is no RP from it -- this is achieved by setting
            # the relative velocities to zero.
            VrelR = 0.
            Vrelphi = 0.
            Vrelz = 0.
        else: # i.e., if component p is not a disk, i.e., a spherical 
            # dark-matter component, we add its contribution
            VrelR = VR
            Vrelphi = Vphi
            Vrelz = Vz
        Vrel = np.sqrt(VrelR**2.+Vrelphi**2.+Vrelz**2.)
        Vrel = max(Vrel,cfg.eps) # safety
        fac_s = p.rho(R,z) * Vrel 
        sR += fac_s * VrelR 
        sphi += fac_s * Vrelphi 
        sz += fac_s * Vrelz 
    return fac*sR, fac*sphi, fac*sz
        
def ftot(potential,xv,m=None,sigmamx=None,Xd=None):
    """
    Total acceleration [(kpc/Gyr)^2 kpc^-1] at phase-space coordinate xv, 
    in an axisymmetric potential. Here "total" means gravitational 
    acceleration plus dynamical-friction acceleration.
    
    Syntax:

        ftot(potential,xv,m=None,sigmamx=None,Xd=None)
        
    where 
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        m: satellite mass [M_sun] (float) 
            (default is None; if provided, dynamical friction is on)
        sigmamx: SIDM cross section [cm^2/g] 
            (default is None; if provided, ram-pressure drag is on)
        Xd: coefficient of ram-pressure deceleration as in Kummer+18
            (default is None; if sigmamx provided, must provide)
    
    Return: 
    
        fR: R-component of total (grav+DF+RP) acceleration (float), 
        fphi: phi-component of total (grav+DF+RP) acceleration (float), 
        fz: z-component of total (grav+DF+RP) acceleration (float)
        
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the total 
    acceleration experienced by a satellite of mass m, self-interaction
    cross-section sigmamx, at location xv, in this combined halo+disk 
    host, we do: 
    
        ftot([halo,disk],xv,m,sigmamx,Xd)
    """
    if not isinstance(potential, list): # if potential is not composite,
        # make it a list of only one element, such that the code below
        # works for both a single potential and a composite potential
        potential = [potential] 
    #
    R, phi, z, VR, Vphi, Vz = xv
    #
    fR,fphi,fz = 0.,0.,0.
    for p in potential:
        fR_tmp, fphi_tmp, fz_tmp = p.fgrav(R,z)
        fR += fR_tmp
        fphi += fphi_tmp
        fz += fz_tmp
    # 
    if m is None: # i.e., if dynamical friction is ignored
        fDFR, fDFphi, fDFz = 0.,0.,0.
    else:
        fDFR, fDFphi, fDFz = fDF(potential,xv,m)
    #
    if sigmamx is None: # i.e., if ram-pressure drag is ignored
        fRPR, fRPphi, fRPz = 0.,0.,0.
    else:
        fRPR, fRPphi, fRPz = fRP(potential,xv,sigmamx,Xd)
    return fR+fDFR+fRPR, fphi+fDFphi+fRPphi, fz+fDFz+fRPz 

def EnergyAngMomGivenRpRa(potential,rp,ra):
    """
    Compute the specific orbital energy [(kpc/Gyr)^2] and the specific 
    orbital angular momentum [kpc(kpc/Gyr)] given two radii along the 
    orbit, e.g., the pericenter and the apocenter radii, using 
    Binney & Tremaine (2008 Eq.3.14)
    
    Syntax:
    
        EnergyAngMomGivenRaRp(potential,rp,ra)
        
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        rp: pericenter radius [kpc] (float)
        ra: apocenter radius [kpc] (float)
    
    Return:
    
        E [(kpc/Gyr)^2], L [kpc(kpc/Gyr)]
    """
    Phip = Phi(potential,rp)
    Phia = Phi(potential,ra)
    upsqr = 1./rp**2
    uasqr = 1./ra**2
    L = np.sqrt(2.*(Phip-Phia)/(uasqr-upsqr))
    E = uasqr/(uasqr-upsqr)*Phip - upsqr/(uasqr-upsqr)*Phia 
    return E,L
    
#---for solving SIDM profile
def f(y,t,a,b):
    """
    Returns right-hand-side functions of the ODEs for solving for the
    dimensionless SIDM potential h(t), as in Kaplinghat+14:
    
        h'(t) = g(t)    
        g'(t) = - [ 2g(t)/t + b/t + a exp(h)/(1-t)^4 ]
        
    where 
    
        t := x/(1+x) with x = r/r_0 is dimensionless radius
        h(t) := - phi(t) / sigma_0^2 is dimensionless potential
        g(t) := h'(t) 
        a := 4 pi G rho_dm0 r_0^2 / sigma_0^2
        b := 4 pi G rho_b0 r_0^2 / sigma_0^2
        
    with rho_dm0, rho_b0 the central DM and baryon density respectively,
    sigma_0 the velocity dispersion in the isothermal core, and r_0 the
    Hernquist scale radius of the baryon distribution.
        
    Syntax:
            
        f(y,t,a,b)
            
    where 
        
        y: [h(t),g(t)] (list or array)
        t: x/(1+x) with x = r/r_0 (float)
        a: 4 pi G rho_dm0 r_0^2 / sigma_0^2 (float)
        b: 4 pi G rho_b0 r_0^2 / sigma_0^2 (float)

    Return: the list of dy/dt:
    
        [g, - (2g+b)/t - a exp(h)/(1-t)^4]
        
    Note that this function shall be used in
    
        scipy.integrate.odeint(f,y0,t,args=(a,b))
    """
    h,g = y
    dydt = [g, -(2.*g+b)/t -a*np.exp(h)/(1.-t)**4]
    return dydt
    
def h(x,a,b):
    """
    Solve for dimensionless SIDM potential profile, h(x), using the 
    Kaplinghat+14 method, given the parameters:
    
        h(x) := - phi(x) / sigma_0^2, x = r/r_0
        a := 4 pi G rho_dm0 r_0^2 / sigma_0^2
        b := 4 pi G rho_b0 r_0^2 / sigma_0^2
    
    with rho_dm0, rho_b0 the central DM and baryon density respectively,
    sigma_0 the velocity dispersion in the isothermal core, and r_0 the
    Hernquist scale radius of the baryon distribution. Note that the SIDM
    density profile is
    
        rho(x) = rho_dm0 exp[h(x)].
        
    Note that for odeint to work, the t variable as in 
        
        odeint(f,[0.,-b/2.],t,args=(a,b))
    
    must be an array and must have the initial-value point as the first 
    element of this array. 
    
    Syntax:
    
        h(x,a,b)
        
    where
    
        x: radius in units of r_0 in ascending order (float or array)
        a: 4 pi G rho_dm0 r_0^2 / sigma_0^2 (float)
        b: 4 pi G rho_b0 r_0^2 / sigma_0^2 (float)
        
    Return: h(x)
    """
    if np.isscalar(x):
        x = np.array([1e-6,x])
    else:
        x = np.append(1e-6,x)
    t = x/(1.+x) 
    sol = odeint(f,[0.,-b/2.],t,args=(a,b))
    return sol[1:,0]

def f2(y,t,a,b):
    """
    Returns right-hand-side functions of the ODEs for solving for the
    dimensionless SIDM potential h(t):
    
        h'(t) = g(t)    
        g'(t) = - b/(1+t)^3 - a exp(h)/t^4
        
    where 
    
        t := r_0/r is a proxy of radius
        h(t) := - phi(t) / sigma_0^2 is dimensionless potential
        g(t) := h'(t) 
        a := 4 pi G rho_dm0 r_0^2 / sigma_0^2
        b := 4 pi G rho_b0 r_0^2 / sigma_0^2
        
    with rho_dm0, rho_b0 the central DM and baryon density respectively,
    sigma_0 the velocity dispersion in the isothermal core, and r_0 the
    Hernquist scale radius of the baryon distribution.
    
    This is a recast of the ODE system in function 'f': while 'f' is for
    integration from center outward, this recast is for integration from
    a boundary radius r_1 inward. 
        
    Syntax:
            
        f2(y,t,a,b)
            
    where 
        
        y: [h(t),g(t)] (list or array)
        t: r_0/r (float)
        a: 4 pi G rho_dm0 r_0^2 / sigma_0^2 (float)
        b: 4 pi G rho_b0 r_0^2 / sigma_0^2 (float)

    Return: the list of dy/dt:
    
        [g, - b/(1+t)^3 - a exp(h)/t^4]
        
    Note that this function shall be used in
    
        scipy.integrate.odeint(f2,y0,t,args=(a,b))
    """
    h,g = y
    dydt = [g, -b/(1.+t)**3 - a*np.exp(h)/t**4]
    return dydt

def h2(x,a,b):
    """
    Solve for dimensionless SIDM potential profile, 
        
        h(x) := - phi(x) / sigma_0^2, x=r/r_0, 
        
    reversely using the Kaplinghat+14 method, i.e., integrating the ODEs 
    from r=r_1 inwards, given the parameters:
    
        a := 4 pi G rho_dm0 r_0^2 / sigma_0^2
        b := 4 pi G rho_b0 r_0^2 / sigma_0^2
    
    with rho_dm0, rho_b0 the central DM and baryon density respectively,
    sigma_0 the velocity dispersion in the isothermal core, and r_0 the
    Hernquist scale radius of the baryon distribution. Note that the SIDM
    density profile is
    
        rho(x) = rho_dm0 exp[h(x)].
        
    Note that this is a variant of the "h" function, and here we 
    integrate from r_1 inwards. For odeint to work, the t variable as in 
        
        odeint(f,y_boundary,t,args=(a,b))
    
    must be an array and must have the boundary-value point r_0/r_1 as 
    the first element of this array. 
    
    Syntax:
    
        h2(x,a,b)
        
    where
    
        x: radius in units of r_0 in ascending order with the 
            last element being x_1 = r_1/r_0 (array)
        a: 4 pi G rho_dm0 r_0^2 / sigma_0^2 (float)
        b: 4 pi G rho_b0 r_0^2 / sigma_0^2 (float)
        
    Return: h(x)
    """
    t = 1./x[::-1]
    pass 

def r1(potential,sigmamx=1.,tage=1.):
    """
    The characteristic radius of a SIDM halo at which an average particle
    is scattered once during the age of the halo, defined as the solution
    to (e.g., Kaplinghat+16 eq.1)
    
        (4/sqrt{pi}) rho(r) sigma(r) (sigma/m_x) t_age = 1
        
    where  Gamma(r) := rho(r) sigma(r) (sigma/m_x) is the scattering rate
    with rho(r) the density profile, sigma(r) the velocity dispersion, 
    and sigma/m_x the self-interaction cross-section per particle mass. 
    
    Syntax:
    
        r1(potential,sigmamx=1.,tage=1.)
        
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        sigmamx: self-interaction cross-section per particle mass 
            [cm^2/g] or [2.08889e-10 kpc^2/M_sun] (default=1.)
        tage: halo age [Gyr], somewhat arbitrary, e.g., lookback time to
            the formation epoch of the halo, where "formation" can be 
            defined as the most recent major merger or the time of 
            reaching half of the current mass.
    """
    a = cfg.Rres
    b = 2000.
    fa = Findr1(a,potential,sigmamx,tage)
    fb = Findr1(b,potential,sigmamx,tage)
    if fa*fb>0.:
        r = cfg.Rres
    else:
        r = brentq(Findr1, a,b, args=(potential,sigmamx,tage),
            xtol=0.001,rtol=1e-5,maxiter=1000)
    return r
def Findr1(r,potential,sigmamx,tage):
    """
    Auxiliary function for the function "r1", which returns the 
        
        left-hand side  -  right-hand side
        
    of the equation 
    
        4/sqrt{pi} rho(r) sigma(r) (sigma/m_x) t_age = 1
    """
    return cfg.FourOverRootPi * rho(potential,r) * sigma(potential,r) *\
        (sigmamx*2.08889e-10) * tage - 1. 
        
def r1_new(potential,sigma0,sigmamx=1.,tage=10.):
    """
    A variant version of the function "r1", where we use the central 
    velocity dispersion sigma_0 in place of the CDM velocity dispersion 
    profile sigma(r).
    
    The characteristic radius of a SIDM halo at which an average particle
    is scattered once during the age of the halo, defined as the solution
    to (e.g., Kaplinghat+16 eq.1)
    
        (4/sqrt{pi}) rho(r) sigma_0 (sigma/m_x) t_age = 1
        
    where  Gamma(r) := rho(r) sigma_0 (sigma/m_x) is the scattering rate
    with rho(r) the density profile, sigma_0 the central vel dispersion, 
    and sigma/m_x the self-interaction cross-section per particle mass. 
    
    Syntax:
    
        r1_new(potential,sigma0,sigmamx=1.,tage=10.)
        
    where
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        sigma0: the central velocity dispersion [kpc/Gyr] (float)
        sigmamx: self-interaction cross-section per particle mass 
            [cm^2/g] or [2.08889e-10 kpc^2/M_sun] (default=1.)
        tage: halo age [Gyr], somewhat arbitrary, e.g., lookback time to
            the formation epoch of the halo, where "formation" can be 
            defined as the most recent major merger or the time of 
            reaching half of the current mass. (default=1.)
    """
    a = cfg.Rres
    b = 2000.
    fa = Findr1_new(a,potential,sigma0,sigmamx,tage)
    fb = Findr1_new(b,potential,sigma0,sigmamx,tage)
    if fa*fb>0.:
        r = cfg.Rres
    else:
        r = brentq(Findr1_new, a,b, args=(potential,sigma0,sigmamx,tage),
            xtol=0.001,rtol=1e-5,maxiter=1000)
    return r
def Findr1_new(r,potential,sigma0,sigmamx,tage):
    """
    Auxiliary function for the function "r1", which returns the 
        
        left-hand side  -  right-hand side
        
    of the equation 
    
        (4/sqrt{pi}) rho(r) sigma_0 (sigma/m_x) t_age = 1
    """
    return cfg.FourOverRootPi * rho(potential,r) * sigma0 *\
        (sigmamx*2.08889e-10) * tage - 1. 
        
def Miso(r,rho):
    """
    Enclosed mass profile of the SIDM isothermal core, given the radii 
    and density profile array.
    
    Syntax:
    
        Miso(r,rho)
        
    where
    
        r: radii at which the density profile is evaluated [kpc] (array)
        rho: density profile [M_sun/kpc^3] (array)
        
    Return: the mass profile corresponding to the input density profile
        [M_sun] (array of the same length as r and rho)
    """
    rtmp = np.append(0.,r[:-1]) 
    dr = r - rtmp
    rhoave = np.append(rho[0],0.5*(rho[1:]+rho[:-1]))
    dM = cfg.FourPi * r**2 *dr * rhoave
    return dM.cumsum()
