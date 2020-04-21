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
from scipy.integrate import quad
from scipy.special import erf,gamma,gammainc,gammaincc 
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
    
        halo = NFW(M,c,Delta=200.,z=0.)
        
    where 
    
        M: halo mass [M_sun], where halo is defined as spherical 
            overdensity of Delta times critical density (float) 
        c: halo concentration (float)
        Delta: average overdensity of the halo, in multiples of the 
            critical density of the Universe (float)
            (default 200.)
        z: redshift (float) (default 0.)
    
    Attributes:
    
        .Mh: halo mass [M_sun]
        .ch: halo concentration
        .Deltah: spherical overdensity wrt instantaneous critical density
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r=sqrt(R^2+z^2)
        .M(R,z=0.): mass [M_sun] enclosed in radius r=sqrt(R^2+z^2)
        .rhobar(R,z=0.): mean density [M_sun kpc^-3] within radius 
            r=sqrt(R^2+z^2)
        .tdyn(R,z=0.): dyn. time [Gyr] within radius r = sqrt(R^2+z^2)
        .Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2)
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z) 
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)      
    
    HISTORY: Arthur Fangzhou Jiang (2016-10-24, HUJI)
             Arthur Fangzhou Jiang (2016-10-30, HUJI)
             Arthur Fangzhou Jiang (2019-08-24, HUJI)
    """
    def __init__(self,M,c,Delta=200.,z=0.):
        """
        Initialize NFW profile.
        
        Syntax:
        
            halo = NFW(M,c,Delta=200.,z=0.)
        
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
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rmax = self.rs * 2.163
        #
        # attributes repeatedly used by following methods
        self.rho0 = self.rhoc*self.Deltah/3.*self.ch**3./self.f(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2.      
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
        return self.Vcirc(self.rmax)*1.4393*x**0.354/(1.+1.1756*x**0.725)
        
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
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: average density of halo [M_sun kpc^-3]
        .rh: halo radius within which density is Delta times rhoc [kpc]
        .rs: scale radius [kpc]
        .rmax: radius at which maximum circular velocity is reached [kpc]
        .sh: logarithmic density slope at 0.01 halo radius
        
    Methods:
    
        .rho(R,z=0.): density [M_sun kpc^-3] at radius r =sqrt(R^2 + z^2)
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
        #
        # derived attributes
        self.rhoc = co.rhoc(z,cfg.h,cfg.Om,cfg.OL)
        self.rhoh = self.Deltah * self.rhoc
        self.rh = (3.*self.Mh / (cfg.FourPi*self.rhoh))**(1./3.)
        self.rs = self.rh / self.ch
        self.rmax = self.rs * (2.-self.alphah)**2.
        self.r2 = self.rmax/2.25
        self.sh = (self.alphah+0.35*self.ch**0.5) / (1.+0.1*self.ch**0.5)
        #
        # attributes repeatedly used by following methods
        self.rho0 = self.rhoc*self.Deltah * (3.-self.alphah)/3. * \
            self.ch**3./self.g(self.ch)
        self.Phi0 = -cfg.FourPiG*self.rho0*self.rs**2. / \
            ((3.-self.alphah)*(2.-self.alphah)*(2.*(2.-self.alphah)+1))
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
        .rhoc: critical density [M_sun kpc^-3]
        .rhoh: halo's average density [M_sun kpc^-3]
        .rh: halo radius [kpc], within which density is Deltah times rhoc
        .rs: scale radius [kpc], at which log density slope is -2
        .nh: inverse of shape paramter (1 / alphah)
        .hh: scale length [kpc], defined as rs / (2/alphah)^(1/alphah)
        .rho0: halo's central density [M_sun kpc^-3]
        .xmax: dimensionless rmax, defined as (rmax/hh)^alphah
        .rmax: radius [kpc] at which maximum circular velocity is reached 
        .Mtot: total mass [M_sun] of the Einasto profile integrated to 
            infinity
        
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
        #
        # supportive attributes
        self.GMtot = cfg.G*self.Mtot 
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
                II = quad(dIdx, xx, np.inf, args=(self.nh,),)[0]
                I.append(II)
            I = np.array(I)
        else:
            I = quad(dIdx, x, np.inf, args=(self.nh,),)[0]
        sigmasqr = self.GMtot/self.hh*self.nh*np.exp(x) * I 
        return np.sqrt(sigmasqr)
def dIdx(x,n):
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
        r = np.sqrt(R**2.+z**2.) 
        return self.Md * r**3. / (r**2.+(self.a+self.b)**2.)**1.5
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
        #.Phi(R,z=0.): potential [(kpc/Gyr)^2] at radius r=sqrt(R^2+z^2) # Not implemented currently, since not needed.
        .fgrav(R,z): grav. acceleration [(kpc/Gyr)^2 kpc^-1] at (R,z)
        .Vcirc(R,z=0.): circ. vel. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
        .sigma(R,z=0.): vel. disp. [kpc/Gyr] at radius r=sqrt(R^2+z^2)
    
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
        """
        # let's make sure that fb > 1e-5
        self.Mh = Mnew
        self.fb = self.Mh / self.Minit
        if(self.fb < cfg.fbv_min):
            self.fb = cfg.fbv_min
            self.Mh = cfg.Mres
            # Note that these won't line up, but this is just to
            # effectively destroy the subhalo if it's been stripped
            # to be less than 10^-5 of its original mass, as it wouldn't
            # show up on SHMFs anyway.
        self.log10fb = np.log10(self.fb)
        return self.Mh # just in case it was set to Mres

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
            type: 'mass' or 'sigma', denoting which profile to compute    
        """

        if(type == 'mass'):
            interp = cfg.fb_cs_interps_mass
            phys_unit_mult = self.Minit
        elif(type == 'sigma'):
            interp = cfg.fb_cs_interps_sigma
            phys_unit_mult = self.sigma0
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

        return val * phys_unit_mult
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
        
def fDF(potential,xv,m):
    """
    Dynamical-friction (DF) acceleration [(kpc/Gyr)^2 kpc^-1] for a 
    satellite of mass m, at phase-space coordinate xv, in an axisymmetric 
    potential:
    
        f_DF = -4pi G^2 m Sum_i rho_i(R,z)F(<|V_i|)ln(Lambda) V_i/|V_i|^3  
    
    where
        
        V_i: relative velocity of the satellite with respect to the host
            component i
        F(<|V_i|) = erf(X) - 2X/sqrt{pi} exp(-X^2)
        X = |V_i| / (sqrt{2} sigma(R,z)) 
        
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
    
        fDF([halo,disk],xv,m,CoulombLogChoice=0)
    
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
        if isinstance(p,(MN,)): # i.e., if potential p is a disk
            lnL = 0.5
            VrelR = VR
            #Vrelphi = Vphi - p.Vcirc(R,z)
            Vrelphi = Vphi - p.Vphi(R,z) # <<< test
            Vrelz = Vz
        else: # i.e., if the potential is a halo
            lnL = np.log(p.Mh/m)
            VrelR = VR
            Vrelphi = Vphi
            Vrelz = Vz
        Vrel = np.sqrt(VrelR**2.+Vrelphi**2.+Vrelz**2.)
        Vrel = max(Vrel,cfg.eps) # safety
        X = Vrel / (cfg.Root2 * p.sigma(R,z))
        fac_s = p.rho(R,z) * lnL * ( erf(X) - \
            cfg.TwoOverRootPi*X*np.exp(-X**2.) ) / Vrel**3 
        sR += fac_s * VrelR 
        sphi += fac_s * Vrelphi 
        sz += fac_s * Vrelz 
    return fac*sR, fac*sphi, fac*sz
        
def ftot(potential,xv,m=None):
    """
    Total acceleration [(kpc/Gyr)^2 kpc^-1] at phase-space coordinate xv, 
    in an axisymmetric potential. Here "total" means gravitational 
    acceleration plus dynamical-friction acceleration.
    
    Syntax:

        ftot(potential,xv,m=None)
        
    where 
    
        potential: host potential (a density profile object, or a list of
            such objects that constitute a composite potential)
        xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] 
            [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        m: satellite mass [M_sun] (float) 
            (default is None; if provided, dynamical friction is on)
    
    Return: 
    
        fR: R-component of total (grav+DF) acceleration (float), 
        fphi: phi-component of total (grav+DF) acceleration (float), 
        fz: z-component of total (grav+DF) acceleration (float)
        
    Example: we have a potential consisting of an NFW halo and a MN disk,
        
        halo = NFW(10.**12,10.,Delta=200,Om=0.3,h=0.7)
        disk = MN(10.**10,6.5,0.25)
    
    i.e., potential = [halo,disk], and we want to get the total 
    acceleration
    experienced by a satellite of mass m at xv in this combined 
    halo+disk host, we do: 
    
        ftot([halo,disk],xv,m)
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
    return fR+fDFR, fphi+fDFphi, fz+fDFz 