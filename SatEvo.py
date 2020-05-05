################################ SatEvo #################################

# Program that evolves the satellites intialized by TreeGen.py

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016-2017 Hebrew University
# Arthur Fangzhou Jiang 2020 Caltech

######################## set up the environment #########################

import config as cfg
import cosmo as co
import evolve as ev
from profiles import NFW,Dekel,MN,Einasto
from orbit import orbit
import galhalo as gh
import aux

import numpy as np
import sys
import os 
import time 
from multiprocessing import Pool, cpu_count

# <<< for clean on-screen prints, use with caution, make sure that 
# the warning is not prevalent or essential for the result
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

########################### user control ################################

fd = 0.1 # <<< play with, disk mass fraction: no disk potential if 0
flattening = 25. # disk scale radius / disk scale height  
fb = 0.0 # <<< play with, bulge mass fraction: no bulge potential if 0

datadir = "./OUTPUT_TREE_MW_NIHAO/" 
#outdir = "./OUTPUT_SAT_MW_fd%.2f_fb%.2f_NIHAO/"%(fd,fb)
#outdir = "./OUTPUT_SAT_MW_fd%.2f_fb%.2f_NIHAO_EnhancedDF/"%(fd,fb)
#outdir = "./OUTPUT_SAT_MW_fd%.2f_fb%.2f_NIHAO_WeakenedDF/"%(fd,fb)
outdir = "./OUTPUT_SAT_MW_fd%.2f_fb%.2f_NIHAO_ZeroDiskDF/"%(fd,fb)
#datadir = "./OUTPUT_TREE_MW_APOSTLE/" 
#outdir = "./OUTPUT_SAT_MW_fd%.2f_fb%.2f_APOSTLE/"%(fd,fb)

#datadir = "./OUTPUT_TREE_GROUP_NIHAO/" 
#outdir = "./OUTPUT_SAT_GROUP_NIHAO/"
#datadir = "./OUTPUT_TREE_GROUP_APOSTLE/" 
#outdir = "./OUTPUT_SAT_GROUP_APOSTLE/"

cfg.Mres = 1e6 # <<< using that in TreeGen.py is enough
cfg.Rres = 0.01 # [kpc] <<< use 0.001 if want to resolve UCDs

StrippingEfficiency = 0.6 # <<< play with

########################### evolve satellites ###########################

#---get the list of data files
files = []    
for filename in os.listdir(datadir):
    if filename.startswith('tree') and filename.endswith('.npz'): 
        files.append(os.path.join(datadir, filename))
files.sort()

#---creating output directory
print('>>> Creating output directory %s ...' % outdir)
try:
    os.mkdir(outdir)
except OSError:
    print ("    Alarm! Creation of the directory %s failed." % outdir)
else:
    print ("    Successfully created the directory %s. " % outdir)
#sys.exit()

print('>>> Evolving satellites ...')

#---
time_start = time.time()
#for file in files: # <<< serial run, only for testing
def loop(file): 
    """
    Replaces the loop "for file in files:", for parallelization.
    """
  
    time_start_tmp = time.time()  
    
    #---load trees
    f = np.load(file)
    redshift = f['redshift']
    CosmicTime = f['CosmicTime']
    mass = f['mass']
    order = f['order']
    ParentID = f['ParentID']
    VirialRadius = f['VirialRadius']
    concentration = f['concentration']
    DekelConcentration = f['DekelConcentration']
    DekelSlope = f['DekelSlope']
    StellarMass = f['StellarMass']
    StellarSize = f['StellarSize']
    coordinates = f['coordinates']
    VirialOverdensity = np.zeros(mass.shape,np.float32) + 200.
    MaxCircularVelocity = np.zeros(mass.shape,np.float32) - 99.
    
    #---identify the roots of the branches
    izroot = mass.argmax(axis=1) # root-redshift ids of all the branches
    idx = np.arange(mass.shape[0]) # branch ids of all the branches
    levels = np.unique(order[order>0]) # all >0 levels in the tree
    
    for level in levels: # loop over levels from low to high
    
        for id in idx: # loop over branches
            
            iza = izroot[id]
            if order[id,iza]!=level: continue # level by level
            
            #---initialize
            
            # read satellite properties at accretion
            za = redshift[iza]
            ta = CosmicTime[iza]
            ma = mass[id,iza]
            ka = order[id,iza]
            ipa = ParentID[id,iza]
            ca = DekelConcentration[id,iza]
            aa = DekelSlope[id,iza]
            c2a = concentration[id,iza]
            Da = VirialOverdensity[id,iza]
            msa = StellarMass[id,iza]
            lea = StellarSize[id,iza]
            xva = coordinates[id,iza,:]
            
            # initialize satellite and orbit
            s = Dekel(ma,ca,aa,Delta=Da,z=za)
            o = orbit(xva)
            
            # initialize quantities repeatedly used for tidal tracks
            s001a = s.sh
            lma = s.rmax
            vma = s.Vcirc(lma)
            lealma = lea / lma
            mma = s.M(lma) 
            
            MaxCircularVelocity[id,iza] = vma 
            
            # initialize instantaneous quantities to be updated
            m = ma
            r = np.sqrt(xva[0]**2+xva[2]**2)
            k = ka 
            ip = ipa
            trelease = ta # cosmic time at lastest release
            tprevious = ta # cosmic time at previous timestep
            
            #---evolve
            for iz in np.arange(iza)[::-1]: # loop over time to evolve
                
                z = redshift[iz]
                tcurrent = CosmicTime[iz]
                
                #---time since in current host
                t = tcurrent - trelease
                
                #---timestep
                dt = tcurrent - tprevious
                
                #---update host potential
                Mp = mass[ip,iz]
                cp = DekelConcentration[ip,iz]
                ap = DekelSlope[ip,iz]
                Dp = VirialOverdensity[ip,iz] 
                if (fd>0.) and (k==1):
                    
                    c2p = concentration[ip,iz]
                    Rvp = VirialRadius[ip,iz]
                    Rep = gh.Reff(Rvp,c2p)
                    adp = 0.766421/(1.+1./flattening) * Rep
                    bdp = adp / flattening
                    Mdp = fd * Mp
                    p = [Dekel(Mp,cp,ap,Delta=Dp,z=z),MN(Mdp,adp,bdp),]
                
                else:
                    
                    p = [Dekel(Mp,cp,ap,Delta=Dp,z=z),]
                    
                #---evolve orbit
                if r>cfg.Rres:
                
                    o.integrate(t,p,m)
                    xv = o.xv # note that the coordinates are updated 
                    # internally in the orbit instance "o" when calling
                    # the ".integrate" method, here we assign them to 
                    # a new variable "xv" only for bookkeeping
                    
                else: # i.e., the satellite has merged to its host, so
                    # no need for orbit integration; to avoid potential 
                    # numerical issues, we assign a dummy coordinate that 
                    # is almost zero but not exactly zero
                
                    xv = np.array([cfg.Rres,0.,0.,0.,0.,0.])
                
                r = np.sqrt(xv[0]**2+xv[2]**2)
                
                #---evolve satellite
                if m>cfg.Mres:
                    
                    # evolve subhalo properties
                    m,lt = ev.msub(s,p,xv,dt,choice='King62',
                        alpha=StrippingEfficiency)
                    a = s.alphah # assume constant innermost slope
                    c,D = ev.Dekel2(m,ma,lma,vma,aa,s001a,z=z) 
                    s = Dekel(m,c,a,Delta=D,z=z)
                    lh = s.rh
                    c2 = s.rh/s.r2
                    s001 = s.sh
                    vm = s.Vcirc(s.rmax)
                    
                    # evolve baryonic properties
                    mm = s.M(s.rmax) # update m_max
                    g_le, g_ms = ev.g_EPW18(mm/mma,s001a,lealma) 
                    le = lea * g_le
                    ms = msa * g_ms
                    ms = min(ms,m) # <<< safety, perhaps rarely triggered
                
                else: # we do nothing about disrupted satellite, s.t.,
                    # its properties right before disruption would be 
                    # stored in the output arrays
                    
                    pass
                
                #---if order>1, determine if releasing this high-order 
                #   subhalo to its grandparent-host, and if releasing,
                #   update the orbit instance
                if k>1:
                
                    if (r > VirialRadius[ip,iz]) & (iz <= izroot[ip]): 
                        # <<< play with the release condition: for now, 
                        # release if the instant orbital radius is larger 
                        # than the virial radius of the immediate host,
                        # and if the immediate host is already a 
                        # satellite of the grandparent-host
                    
                        xv = coordinates[ip,iz,:] # <<< may change later:
                            # for now, the released satellite picks up 
                            # the coordinates of its immediate parent
                        o = orbit(xv)
                        ip = ParentID[ip,iz] # update parent id
                        k = k - 1 # update instant order
                        trelease = tcurrent # update release time
                
                #---update the arrays for output
                mass[id,iz] = m 
                order[id,iz] = k
                ParentID[id,iz] = ip
                VirialRadius[id,iz] = lh
                concentration[id,iz] = c2
                DekelConcentration[id,iz] = c
                DekelSlope[id,iz] = a
                VirialOverdensity[id,iz] = D
                MaxCircularVelocity[id,iz] = vm
                StellarMass[id,iz] = ms
                StellarSize[id,iz] = le
                coordinates[id,iz,:] = xv
                
                #---update tprevious
                tprevious = tcurrent
            
                # <<< test
                #print('    id=%5i,k=%2i,ip=%5i,z=%6.2f,r=%9.4f,log(m)=%6.2f,D=%7.1f,c2=%6.2f,s001=%5.2f,log(ms)=%6.2f,le=%7.2f,xv=%7.2f,%7.2f,%7.2f,%7.2f,%7.2f,%7.2f'%\
                #    (id,k,ip,z,r,np.log10(m),D,c2,s001,np.log10(ms),le, xv[0],xv[1],xv[2],xv[3],xv[4],xv[5]))
    
    #---output
    outfile = outdir + file[len(datadir):]
    np.savez(outfile, 
        redshift = redshift,
        CosmicTime = CosmicTime,
        mass = mass,
        order = order,
        ParentID = ParentID,
        VirialRadius = VirialRadius,
        concentration = concentration,
        DekelConcentration = DekelConcentration,
        DekelSlope = DekelSlope,
        VirialOverdensity = VirialOverdensity,
        MaxCircularVelocity = MaxCircularVelocity,
        StellarMass = StellarMass,
        StellarSize = StellarSize,
        coordinates = coordinates,
        )
    
    #---on-screen prints
    M0 = mass[0,0]
    m0 = mass[:,0][1:]
    Ms0 = StellarMass[0,0]
    ms0 = StellarMass[:,0][1:]
    
    msk = (m0 > 1e-4*M0) & (m0 < M0) # <<< latter condition to be removed 
    fsub = m0[msk].sum() / M0
    fstar = ms0[msk].sum() / Ms0
    
    MAH = mass[0,:]
    iz50 = aux.FindNearestIndex(MAH,0.5*M0)
    z50 = redshift[iz50]
    
    time_end_tmp = time.time()
    print('    %s: %5.2f min, z50=%5.2f,fsub(>1e-4)=%8.5f,fstar=%8.5f'%\
        (outfile,(time_end_tmp-time_start_tmp)/60.,z50,fsub,fstar))
    #sys.exit() # <<< test

#---for parallelization, comment for testing in serial mode
if __name__ == "__main__":
    pool = Pool(cpu_count()) # use all cores
    pool.map(loop, files)

time_end = time.time() 
print('    total time: %5.2f hours'%((time_end - time_start)/3600.))
