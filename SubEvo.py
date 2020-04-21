################################ SubEvo #################################

# Program that evolves the subhaloes intialized by TreeGen_Sub.py
# This version of the code is meant to work with the Green model of
# stripped subhalo density profiles.

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016-2017 Hebrew University
# Arthur Fangzhou Jiang 2020 Caltech
# Sheridan Beckwith Green 2020 Yale University

######################## set up the environment #########################

import config as cfg
import cosmo as co
import evolve as ev
from profiles import NFW,Green
from orbit import orbit
import aux

import numpy as np
import sys
import os 
import time 
from multiprocessing import Pool, cpu_count

# <<< for clean on-screen prints, use with caution, make sure that 
# the warning is not prevalent or essential for the result
import warnings
#warnings.simplefilter('always', UserWarning)
#warnings.filterwarnings("ignore", category=RuntimeWarning) 
warnings.simplefilter("ignore", UserWarning)
# TODO: Look at warnings

########################### user control ################################

datadir = "./OUTPUT_TREE/" 
outdir = "./OUTPUT_SAT/"

cfg.Mres = 1e8 # <<< using that in TreeGen.py is enough
cfg.Rres = 0.1 # [kpc] <<< use 0.001 if want to resolve UCDs

# TODO: Add a flag for the stripping efficiency version (0.55 vs. w/ c-dependence)
# TODO: Add a flag for the version of dynamical friction that we use. (just make it a prefactor stored in the config file)

########################### evolve satellites ###########################

#---get the list of data files
files = []    
for filename in os.listdir(datadir):
    if filename.startswith('tree') and filename.endswith('.npz'): 
        files.append(os.path.join(datadir, filename))
files.sort()

print('>>> Evolving subhaloes ...')

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
    coordinates = f['coordinates']

    # compute the virial overdensities for all redshifts
    VirialOverdensity = co.DeltaBN(redshift, cfg.Om, cfg.OL) # same as Dvsample
    GreenRte = np.zeros(VirialRadius.shape) - 99. # contains r_{te} values
    alphas = np.zeros(VirialRadius.shape) - 99.
    tdyns  = np.zeros(VirialRadius.shape) - 99.

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
            Dva = VirialOverdensity[iza]
            ma = mass[id,iza] # initial mass that we will use for f_b
            ka = order[id,iza]
            ipa = ParentID[id,iza]
            c2a = concentration[id,iza]
            xva = coordinates[id,iza,:]
            
            # initialize satellite and orbit
            s = Green(ma,c2a,Delta=Dva,z=za)
            o = orbit(xva) # Make sure to change orbit init in TreeGen
            xv = o.xv
            
            # initialize instantaneous quantities to be updated
            m = ma
            m_old = ma
            r = np.sqrt(xva[0]**2+xva[2]**2)
            k = ka 
            ip = ipa
            trelease = ta # cosmic time at lastest release
            #tprevious = ta # cosmic time at previous timestep
            tcurrent = ta # cosmic time at current timestep

            #---evolve
            #for iz in np.arange(iza)[::-1]: # loop over time to evolve
            for iz in np.arange(iza, 0, -1): # loop over time to evolve
                # NOTE: we now start at iza, rather than iza-1
                # NOTE: we now stop at 1, rather than 0

                iznext = iz - 1                
                z = redshift[iz]
                #tcurrent = CosmicTime[iz]
                tnext = CosmicTime[iznext]
                
                #---time since in current host
                #t = tcurrent - trelease
                t = tnext - trelease

                #---timestep
                #dt = tcurrent - tprevious
                dt = tnext - tcurrent
                
                #---update host potential
                Mp  = mass[ip,iz]
                if(iz >= izroot[ip]):
                    c2p = concentration[ip,iz]
                    Dp  = VirialOverdensity[iz]
                    if(iz == izroot[ip]): # final point before accretion
                        # NOTE: This is the last redshift that concentration
                        # and virial radius are computed, as they become
                        # meaningless afterwards once the host profile
                        # begins to be stripped.
                        p   = Green(Mp,c2p,Delta=Dp,z=z)
                    else: # host hasn't been accreted yet, just grow NFW
                        p   = NFW(Mp,c2p,Delta=Dp,z=z)
                else: # host is itself a subhalo, update mass and f_b
                    p.update_mass(Mp)

               
                # compute alpha for matrix
                alpha = ev.alpha_from_c2(p.ch, s.ch)
                
                #---evolve satellite
                if m>cfg.Mres:
                    
                    # evolve subhalo properties
                    m,lt = ev.msub(s,p,xv,dt,choice='King62',
                        alpha=alpha)
                    m = s.update_mass(m)
                    rte = s.rte()
                    
                
                else: # we do nothing about disrupted satellite, s.t.,
                    # its properties right before disruption would be 
                    # stored in the output arrays
                    
                    pass
                
                #---evolve orbit
                if r>cfg.Rres:
                
                    tdyn = p.tdyn(r)
                    o.integrate(t,p,m_old)
                    xv = o.xv # note that the coordinates are updated 
                    # internally in the orbit instance "o" when calling
                    # the ".integrate" method, here we assign them to 
                    # a new variable "xv" only for bookkeeping
                    #print("just after integrate", np.sqrt(xv[0]**2+xv[2]**2))
                    
                else: # i.e., the satellite has merged to its host, so
                    # no need for orbit integration; to avoid potential 
                    # numerical issues, we assign a dummy coordinate that 
                    # is almost zero but not exactly zero
                    tdyn = p.tdyn(cfg.Rres)
                
                    xv = np.array([cfg.Rres,0.,0.,0.,0.,0.])

                r = np.sqrt(xv[0]**2+xv[2]**2)
                m_old = m


                #---if order>1, determine if releasing this high-order 
                #   subhalo to its grandparent-host, and if releasing,
                #   update the orbit instance
                if k>1:
                
                    if (r > VirialRadius[ip,iz]) & (iz <= izroot[ip]): 
                        # <<< Release condition:
                        # 1. Host halo is already within a grandparent-host
                        # 2. Instant orbital radius is larger than the host
                        # TIDAL radius (note that VirialRadius also contains
                        # the tidal radii for the host haloes once they fall
                        # into a grandparent-host)
                        # 3. (below) We compute the fraction of:
                        #             alpha * dynamical time
                        # corresponding to this dt, and release with
                        # probability dt / (alpha * dynamical time) 
                        # TODO: Consider how to deal with this criteria
                        # integrated over multiple timesteps outside of
                        # the tidal radius..

                        # Compute probability of being ejected
                        odds = np.random.rand()
                        dyntime_frac = dt / (alphas[ip,iz] * tdyns[ip,iz])
                        if(odds < dyntime_frac):
                            if(ParentID[ip,iz] == ParentID[ip,iznext]):
                                # host wasn't also released at same time
                                # New coordinates at next time are the
                                # updated subhalo coordinates plus the updated
                                # host coordinates inside of grandparent
                                xv = aux.add_cyl_vecs(xv,coordinates[ip,iznext,:])
                                # NOTE: Changed this to pick up the 
                                # orbital coordinates of the host at the
                                # next timestep PLUS the orbital coordinates
                                # of the sub w.r.t. the host
                            else:
                                xv = aux.add_cyl_vecs(xv,coordinates[ip,iz,:])
                                # This will be extraordinarily rare, but just
                                # a check in case so that the released order-k
                                # subhalo isn't accidentally double-released
                                # in terms of updated coordinates, but not
                                # in terms of new host ID.
                            o = orbit(xv)
                            k = order[ip,iz] # update instant order to the same as the parent

                            ip = ParentID[ip,iz] # update parent id
                            izp = izroot[ip]
                            Mp = mass[ip,izp]
                            c2p = concentration[ip,izp]
                            Dp  = VirialOverdensity[izp]
                            p = Green(Mp,c2p,Delta=Dp,z=redshift[izp])
                            # NOTE: Mass will be updated next time, so don't
                            # need to do it yet
                            # Also, if new host hasn't merged yet, it will be
                            # converted to an NFW at start of next step.
                            #trelease = tcurrent # update release time
                            trelease = tnext # update release time
                
                #---update the arrays for output
                mass[id,iznext] = m 
                order[id,iznext] = k
                ParentID[id,iznext] = ip
                VirialRadius[id,iznext] = lt # storing tidal radius
                # NOTE: We won't be storing concentrations
                # NOTE: We store tidal radius in lieu of virial radius
                # for haloes after they start getting stripped
                GreenRte[id,iznext] = rte
                coordinates[id,iznext,:] = xv

                # note that the below two are quantities
                # at current timestep instead, since only used for
                # host release criteria
                # This won't be output since only used internally
                alphas[id,iz] = alpha
                tdyns[id,iz] = tdyn
                
                #---update tprevious
                #tprevious = tcurrent
                tcurrent = tnext
            
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
        GreenRte = GreenRte,
        # this contains values during stripping, -99 prior to stripping and 
        # once the halo reaches Mres, i.e. "disrupted" -- 
        # for visualization purposes
        concentration = concentration, # this is unchanged from TreeGen output
        coordinates = coordinates,
        )
    
    #---on-screen prints
    M0 = mass[0,0]
    m0 = mass[:,0][1:]
    
    msk = (m0 > 1e-4*M0) & (m0 < M0) # <<< latter condition to be removed 
    fsub = m0[msk].sum() / M0
    
    MAH = mass[0,:]
    iz50 = aux.FindNearestIndex(MAH,0.5*M0)
    z50 = redshift[iz50]
    
    time_end_tmp = time.time()
    print('    %s: %5.2f min, z50=%5.2f,fsub=%8.5f'%\
        (outfile,(time_end_tmp-time_start_tmp)/60., z50,fsub))
    #sys.exit() # <<< test

#---for parallelization, comment for testing in serial mode
if __name__ == "__main__":
    pool = Pool(cpu_count()) # use all cores
    pool.map(loop, files)

time_end = time.time() 
print('    total time: %5.2f hours'%((time_end - time_start)/3600.))
