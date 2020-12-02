############################# orbit class ###############################

# Arthur Fangzhou Jiang 2016, HUJI --- original version

# Arthur Fangzhou Jiang 2019, HUJI, UCSC --- revisions:
# - no longer convert speed unit from kpc/Gyr to km/s 
# - improved dynamical-friction (DF) (see profiles.py for more details)

#########################################################################

from profiles import ftot

import numpy as np
from scipy.integrate import ode

import warnings
#warnings.filterwarnings("ignore", category=UserWarning)

#########################################################################

#---
class orbit(object):
    """
    Class for orbit and orbit integration in axisymetric potential
    
    Syntax:
    
        o = orbit(xv,potential=None)
        
    which initializes an orbit object "o", where
    
        xv: the phase-space coordinate in a cylindrical frame 
            [R,phi,z,VR,Vphi,Vz] [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr]
            (numpy array)
        potential: host potential (a density profile object, as defined 
            in profiles.py, or a list such objects that constitute a 
            composite potential)
        
    Attributes:
    
        o.t: integration time [Gyr] (float, list, or array)  
        o.xv: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            which are initialized to be the value put in by hand when 
            the orbit object is created, and are updated once the method 
            o.integrate is called
            (numpy array)
        o.tList: time sequence
        o.xvList: coordinates along the time sequence
    
    and conditionally also: (only available if potential is not None and 
    spherically symmetric) 
    
        o.rperi: peri-center distance [kpc] (float)
        o.rapo: apo-center distance [kpc] (float)
        
    Methods:
    
        o.integrate(t,potential,m=None,CoulombLogChoice=None):
            updates the coordinates o.xv by integrates over time "t" in 
            the "potential", using scipy.integrate.ode, and considering 
            dynamical friction if "m" and "CoulombLogChoice" are provided 
        
    Arthur Fangzhou Jiang, 2016-10-27, HUJI
    Arthur Fangzhou Jiang, 2019-08-21, UCSC
    """
    def __init__(self,xv,potential=None):
        r"""
        Initialize an orbit by specifying the phase-space coordinates 
        in a cylindrical frame.
        
        Syntax:
            
            o = orbit(xv, potential=None)
        
        where 
        
            xv: phase-space coordinates in a cylindrical frame
                [R,phi,z,VR,Vphi,Vz] 
                [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
                (numpy array)
            potential: host potential (a density profile object, as 
                defined in profiles.py, or a list such objects that 
                constitute a composite potential)
                (dafault=None, i.e., when initializing an orbit, do not
                specify the potential, but if provided, a few more 
                attributes attributes are triggered, including o.rperi
                and o.rapo, and maybe more, to be determined)
        """
        self.xv = xv # instantaneous coordinates, initialized by input
        self.t = 0. # instantaneous time, initialized here to be 0. 
        self.tList = [] # initialize time sequencies
        self.xvList = []
        if potential is not None: 
            pass # <<< to be added: self.rperi and self.rapo etc 
    def integrate(self,t,potential,m=None):
        r"""
        Integrate orbit over time "t" [Gyr], using methods that comes 
        with scipy.integrate.ode; and update coorinates to be the values 
        at the end of t.
        
        Syntax:
        
            o.integrate(t,potential,m=None)
        
        where
        
            t: time [Gyr] (float, list, or numpy array)
            potential: host potential (a profile object, as defined 
                in profile.py, or a list of such objects which 
                altogether constitute the host potential)
            m: satellite mass [Msun] 
                (default is None; if provided, dynamical friction is 
                triggered)
                
        Note that in case when t is list or array, attributes such as 
        
            .tList
            .xvList
        
        which are lists, will start from empty and get appended new 
        value for each timestep; while attributes
        
            .t
            .xv
            
        store the instantaneous time and coordinates atthe  end of t.     
        """
        solver = ode(f,jac=None).set_integrator(
            #'vode', 
            'dopri5',
            nsteps=500, # default=500
            max_step = 0.1, # default=0.0 
            rtol=1e-5, # default = 1e-6
            atol=1e-5, # default = 1e-12
            )
        solver.set_initial_value(self.xv, self.t)
        solver.set_f_params(potential,m,)            
        if isinstance(t, list) or isinstance(t,np.ndarray): 
            for tt in t:               
                solver.integrate(tt)
                self.t = tt
                self.xv = solver.y
                self.tList.append(self.t) 
                self.xvList.append(self.xv)
        else: # i.e., if t is a scalar
            solver.integrate(t)
            self.xv = solver.y
            self.t = solver.t
            self.tList.append(self.t) 
            self.xvList.append(self.xv)
        self.tArray = np.array(self.tList)
        self.xvArray = np.array(self.xvList)

def f(t,y,p,m):
    r"""
    Returns right-hand-side functions of the EOMs for orbit integration:
    
        d R / d t = VR    
        d phi / d t = Vphi / R
        d z / d t = Vz
        d VR / dt = Vphi^2 / R + fR 
        d Vphi / dt = - VR * Vphi / R + fphi
        d Vz / d t = fz
        
    for use in the method ".integrate" of the "orbit" class.
        
    Syntax:
            
        f(t,y,p,m)
            
    where 
        
        t: integration time [Gyr] (float)
        y: phase-space coordinates in a cylindrical frame
            [R,phi,z,VR,Vphi,Vz] [kpc,radian,kpc,kpc/Gyr,kpc/Gyr,kpc/Gyr] 
            (numpy array)
        p: host potential (a density profile object, as defined 
            in profiles.py, or a list of such objects that constitute a 
            composite potential)
        m: satellite mass [Msun] (float or None) 
            (If m is None, DF is ignored)

    Note that fR, fphi, fz are the R-, phi- and z-components of the 
    acceleration at phase-space location y, computed by the function 
    "ftot" defined in profiles.py. 

    Return: the list of
    
        [VR,    
         Vphi / R
         Vz,
         Vphi^2 / R + fR ,
         - VR * Vphi / R + fphi,
         fz]

    i.e., the right-hand side of the EOMs describing the evolution of the
    phase-space coordinates in a cylindrical frame
    """
    R, phi, z, VR, Vphi, Vz = y
    fR, fphi, fz = ftot(p,y,m)
    R = max(R,1e-6) # safety
    return [VR, 
        Vphi/R, 
        Vz,
        Vphi**2./R + fR,
        - VR*Vphi/R + fphi,
        fz]
