#########################################################################

# Auxiliary functions.

# Arthur Fangzhou Jiang 2019 Hebrew University

#########################################################################

import config as cfg

import numpy as np
import sys
from fast_histogram import histogram2d

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
    
        ysample: the y values at the x coordinates closest to xgrid,
        xsample: subset of xgrid that corresponds to the range of x
    
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
    r"""
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