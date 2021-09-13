#########################################################################

# Auxiliary functions.

# Arthur Fangzhou Jiang 2019 Hebrew University
# Sheridan Beckwith Green 2020 Yale University

#########################################################################

import config as cfg

import numpy as np
import sys
from fast_histogram import histogram2d
from scipy.integrate import quad

#########################################################################

#---for getting the size of an array

def memory(x):
    """
    memory size of an object in MB
    
    Syntax:
    
        memory(x)
        
    where
    
        x: a python object 
    """
    return sys.getsizeof(x)/1048576.

#---for plotting multi-color lines

def segments(x, y):
    """
    Create line segments from x and y coordinates, in the correct 
    format for LineCollection (https://stackoverflow.com/questions/
    36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar).
    
    Syntax:
        
        segments(x, y)
    
    where
    
        x:
        y:
        
    Return: 
        
        an array of the shape numlines x (points per line) x 2 (x
    and y) array
    """
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis=1)
    
#---for finding the element in an array nearest to a given value

def FindNearestElement(arr,value):
    r"""
    Returns the element in an array "arr" that is closest to a 
    scalar "value".
    
    Syntax: 
    
        FindNearestElement(arr,value)
    
    where
    
        arr: the array being analyzed (array)
        value: the value of interest (float)
    """
    idx = FindNearestIndex(arr,value)
    return arr[idx]
def FindNearestIndex(arr,value):
    r"""
    Returns the index of the element in an array "arr" that is 
    closest to a scalar "value"
    
    Syntax: 
    
        FindNearestIndex(arr,value)
        
    where
    
        arr: the array being analyzed (array)
        value: the value of interest (float)
    """
    return (np.abs(arr-value)).argmin() 

def FindSignChangeIndex(arr):
    """
    Find the indices of the location of sign changes in an array
    
    Syntax:
    
        FindSignChangeIndex(arr)
        
    where
    
        arr: the array being analyzed (1D or 2D array)
        
    If arr is 2D, return the sign-changing indices in the last axis.
    
    # <<< still need to be polished, as of 2021-07-16 
    """
    return np.where(np.diff(np.sign(arr)))[0]

def downsample(y,x,xgrid):
    """
    Downsample the fine-grid measurements y(x) onto a coarser xgrid.
    
    Syntax:
    
        downsample(y,x,xgrid)
        
    where
    
        y: fine-grid measurements, e.g., mass history M(z) on a fine-step
            redshift sequence z (array)
        x: fine-grid coordinates where the measurements are carried out,
            e.g., the reshift array, z, at which the mass history are
            drawn (array)
        xgrid: the coarse-grid coordinates onto which we downsample the
            fine-grid measurements y(x) (array)
            
    Return 
    
        ysample: the y values at the x coordinates closest to xgrid
            (array),
        xsample: subset of xgrid that corresponds to the range of x
            (array)
            
    Note that because of the dependence on FindClosestIndices, y, x and 
    xgrid all have to be numpy arrays, and x and xgrid have to be sorted
    -- one can regard x and xgrid as the redshift grid of a merger tree.
    """
    idx1 = FindNearestIndex(xgrid,x[0])
    idx2 = FindNearestIndex(xgrid,x[-1])
    xsample = xgrid[idx1:idx2+1] # select the subset of xgrid that 
        # corresponds to the range of array x
    xsample = xsample[xsample>=x[0]] # a safety measure that deals with 
        # the cases where the first element of x, x[0], is higher than 
        # the closest xgrid element, xgrid[idx1] -- i.e., make sure that
        # xsample is the subset of xgrid, such that the first element of 
        # xsample is higher than the first element of the fine-grid x.
    if len(xsample)==0: 
        xsample = xgrid[idx1+1] # a safety measure that deals with the 
        # rare cases where idx1 and idx2 are the same -- this can happen
        # when all the elements of x are close to a single xgrid element.
    idx = FindClosestIndices(x,xsample)
    ysample = y[idx]
    if np.isscalar(xsample): # safety, make sure to return arrays
        xsample = np.array([xsample,])
        ysample = np.array([ysample,])
    return ysample,xsample
def FindClosestElements(arr, values):
    """
    Returns the elements in a numpy array `arr' (which has been sorted) 
    that are closest to the values in the list/array `values', e.g.,
    
        arr = np.arange(0., 20.)
        values = [-2., 100., 2., 2.4, 2.5, 2.6]
        FindClosestElements(arr, values)
    
    which gives
    
        array([ 0., 19.,  2.,  2.,  3.,  3.])
    
    Note: this is based on http://stackoverflow.com/questions/8914491/
    finding-the-nearest-value-and-return-the-index-of-array-in-python
    
    Syntax:
    
        FindClosestElements(arr, values)
        
    where
    
        arr: a sorted numpy array, which, in practice for example, is 
            the redshift grid on which we store merger trees (array)
        values: the target array or list, which, in practice for example,
            is the fine-timestep mass history, M(z), which we downsample
            onto the coarse-timestep redshift grid for output.  
    """
    idx = arr.searchsorted(values)
    idx = np.clip(idx, 1, len(arr)-1)
    left = arr[idx-1]
    right = arr[idx]
    idx -= values - left < right - values
    return arr[idx]
def FindClosestIndices(arr, values):
    """
    Similar to FindClosestElements, but returns the indices.
    """
    idx = arr.searchsorted(values)
    idx = np.clip(idx, 1, len(arr)-1)
    left = arr[idx-1]
    right = arr[idx]
    idx -= values - left < right - values
    return idx

#---

def split(arr,msk):
    """
    Returns subarrays of an arr that obeys certain criteria.
    
    Syntax:
    
        split(arr,msk)
        
    where
        
        arr: the full array
        msk: the boolean mask array of the same length as arr
             
    Return:

        subarrays of "arr" where the boolean "msk" is true
    """
    idx = np.where(msk==False)[0]
    arrtmp = arr.copy()
    arrtmp[idx] = -99
    SubArrList = np.split(arrtmp,idx)
    result = []
    for sub in SubArrList:
        if -99 not in sub:
            result.append(sub)
        elif sub[0]==-99 and sub.size>1:
            result.append(sub[1:])
        elif sub[0]==-99 and sub.size==1:
            continue
        else:
            pass
    return result

#---for vector operations

def normalize(v):
    """ 
    Normalize a vector.  
    
    Syntax:
    
        normalize(v)
    
    where
        
        v: the vector to be normalized (array)
        
    Return:
        
        the unit vector of v
    """
    return v / (v**2).sum()**0.5

def angle(v1, v2):
    """ 
    Angle in radians between two vectors.
    
    Syntax:
    
        angle(v1, v2)
        
    where
    
        v1: vector 1 (array)
        v2: vector 2 (array, must be of the same dimension as v1)
    """
    v1_unit = normalize(v1)
    v2_unit = normalize(v2)
    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1.0, 1.0))

def perpendicular(LOS):
    """
    The unit vectors perpendicular to the line-of-sight vector.
    
    Syntax:
    
        perpendicular(LOS)
    
    where
    
        LOS: unnormalized line-of-sight vector (float array of length 3)
        
    Return:
    
        - eX: unit vector of X (float array of length 3)
        - eY: unit vector of Y (float array of length 3)
        - eLOS: unit vector of the LOS (float array of length 3)
    """
    eLOS = normalize(LOS)
    eX = np.cross(np.random.random(3), eLOS)
    eX = normalize(eX)
    eY = np.cross(eLOS,eX)
    return eX,eY,eLOS
    
def project(rr,eX,eY):
    """
    Project along line-of-sight to the 2D plane defined by the unit 
    vectors "ex" and "ey". 
    
    
    Syntax:
        
        project(rr,eX,eY)
        
    where
    
        rr: the 3D position vectors ( Nx3 array ) 
        eX: unit X vector (float array of length 3)
        eY: unit Y vector (float array of length 3)
    
    Return:
     
        No return; but update the global variables, cfg.RR, which stores 
        2D (X,Y) coordinates of the particles of interest.
    
    Note that for 10^5 particles, a call of this function takes ___ us.
    """
    cfg.RR = np.vstack((
        np.dot(rr,eX),
        np.dot(rr,eY),
        )).T
        
def pixelize(R):
    """
    Pixelize an area of 2R x 2R on the projection plane; bin particles
    into the pixels according to their 2D coordinates.
    
    Syntax:
        pixelize(R)
    where
        R: spatial scale of interest [kpc] (scalar)
        
    Note that, for imshow to work porperly, return the transposed image! 
    
    Note that the coordinates of the pixel edges range from -R to R, 
    and the number of pixels is defined by cfg.Npixel (assuming a
    square image).
    
    Note that this function uses the 2D coordinates in global 2D array 
    cfg.RR, as well as the weights in global array cfg.W. 
    """
    im = histogram2d(cfg.RR[:,0],cfg.RR[:,1],
        range=[[-R, R], [-R, R]],
        bins=[cfg.Npixel,cfg.Npixel],
        weights=cfg.weight)
    PixelArea = ( R / float(cfg.Npixel) )**2.
    im = im / PixelArea
    #im[im==0.] = cfg.Sigma_sky # <<< play with
    im[im==0.] = 1e-6 * im.max()
    return im.T
    
#---temporarily put here, for the program test_SIDMprofiles_fit.py
def slope(r,r_grid,rho_grid):
    """
    Logarithmic slope of density profile at a given radius.
    
    Syntax:
        
        slope(r,r_grid,rho_grid)
        
    where
    
        r:  radius at which we evaluate the slope [kpc] (float)
        r_grid: radius array of the density profile [kpc] (array)
        rho_grid: density profile [M_sun kpc^-3] (array)
    """
    if r<r_grid.min():
        sys.exit('Radius too small. Stop.')
    if r>r_grid.max():
        sys.exit('Radius too large. Stop.')
    i = FindNearestIndex(r_grid,r)
    if i==(len(r_grid)-1): 
        i = i-1
    return - np.log(rho_grid[i+1]/rho_grid[i]) / np.log(r_grid[i+1]/r_grid[i])

def mass(r,r_grid,rho_grid,lnrho_interp):
    """
    Enclosed mass at a given radius, given the (non-parametric) density 
    profile. 
    
    Syntax:
    
        mass(r,r_grid,rho_grid)
        
    where
    
        r: radius at which we evaluate the enclosed mass (float)
        r_grid: radii array of the density profile [kpc] (array)
        rho_grid: density profile [M_sun kpc^-3] (array)
        lnrho_interp: interpolation function ln(rho) as a function of 
            ln(r), based on the density profile grid rho_grid and r_grid
    """ 
    if r<r_grid.min():
        sys.exit('Radius too small. Stop.')
    if r>r_grid.max():
        sys.exit('Radius too large. Stop.')
    f = lambda lnr: cfg.FourPi* (np.exp(lnr))**3 * np.exp(lnrho_interp(lnr))
    I = quad(f, np.log(r_grid[0]), np.log(r), args=(), 
        epsabs=1.e-7, epsrel=1.e-6,limit=10000)[0]
    return cfg.FourPi/3.*r_grid[0]**3 * rho_grid[0] + I 

def add_cyl_vecs(xv1, xv2):
    """
    Given two 6D position+velocity vectors in the cylindrical coordinate
    system, computes their vector sum and returns a new 6D position+
    velocity vector.
    
    Syntax:
        
        add_cyl_vecs(xv1, xv2)
        
    where
    
        xv1: the first 6D position+velocity vector (float array of length 6)
        xv2: the second 6D position+velocity vector (float array of length 6)
    
    Return:
     
        xvnew: the vector sum of xv1 and xv2 (float array of length 6)
    """
    R1, phi1, z1, VR1, Vphi1, Vz1 = xv1
    R2, phi2, z2, VR2, Vphi2, Vz2 = xv2
    xvnew = np.zeros(6)
    xvnew[2] = z1 + z2 # z add directly
    xvnew[5] = Vz1 + Vz2
    xnew = R1*np.cos(phi1) + R2*np.cos(phi2)
    ynew = R1*np.sin(phi1) + R2*np.sin(phi2)
    Rnew = np.sqrt(xnew**2. + ynew**2.)
    phinew = np.arctan2(ynew, xnew)
    xvnew[0] = Rnew
    xvnew[1] = phinew
    Vx1 = np.cos(phi1)*VR1 - np.sin(phi1)*Vphi1
    Vy1 = np.sin(phi1)*VR1 + np.cos(phi1)*Vphi1
    Vx2 = np.cos(phi2)*VR2 - np.sin(phi2)*Vphi2
    Vy2 = np.sin(phi2)*VR2 + np.cos(phi2)*Vphi2
    Vxnew = Vx1 + Vx2
    Vynew = Vy1 + Vy2
    VRnew = np.cos(phinew)*Vxnew + np.sin(phinew)*Vynew
    Vphinew = -np.sin(phinew)*Vxnew + np.cos(phinew)*Vynew
    xvnew[3] = VRnew
    xvnew[4] = Vphinew
    return xvnew
