################################ SubEvo #################################

# Program that evolves the subhaloes intialized by TreeGen_Sub.py
# This version of the code is meant to work with the Green model of
# stripped subhalo density profiles.

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016-2017 Hebrew University
# Arthur Fangzhou Jiang 2020 Caltech
# Sheridan Beckwith Green 2020 Yale University
# -- Changed loop order so that redshift is the outermost loop,
#    which enables mass of ejected subhaloes to be removed from
#    the corresponding host; necessary for mass conservation

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
warnings.simplefilter("ignore", UserWarning)

########################### user control ################################


datadir = "./OUTPUT_TREE/"
outdir = "./OUTPUT_SAT/"

Rres_factor = 10**-3 # (Defunct)

#---stripping efficiency type
alpha_type = 'conc' # 'fixed' or 'conc'

#---dynamical friction strength
cfg.lnL_pref = 0.75 # Fiducial, but can also use 1.0

#---evolution mode (resolution limit in m/m_{acc} or m/M_0)
cfg.evo_mode = 'arbres' # or 'withering'

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

    # skip if we already ran this one and are re-running
    # uncompleted trees on a second pass-through
    outfile = outdir + file[len(datadir):]
    if(os.path.exists(outfile)):
        # NOTE: This will throw error if serial
        # Change the below to "continue" for serial
        return
        #continue

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
    levels = np.unique(order[order>=0]) # all >0 levels in the tree
    izmax = mass.shape[1] - 1 # highest redshift index

    #---get smallest host rvir from tree
    #   Defunct, we no longer use an Rres; all subhaloes are evolved
    #   until their mass falls below resolution limit
    min_rvir = VirialRadius[0, np.argwhere(VirialRadius[0,:] > 0)[-1][0]]
    cfg.Rres = min(0.1, min_rvir * Rres_factor) # Never larger than 100 pc

    #---list of potentials and orbits for each branch
    #   additional, mass of ejected subhaloes stored in ejected_mass
    #   to be removed from corresponding host at next timestep
    potentials = [0] * mass.shape[0]
    orbits = [0] * mass.shape[0]
    trelease = np.zeros(mass.shape[0])
    ejected_mass = np.zeros(mass.shape[0])

    #---list of minimum masses, below which we stop evolving the halo
    M0 = mass[0,0]
    min_mass = np.zeros(mass.shape[0])

    #---evolve
    for iz in np.arange(izmax, 0, -1): # loop over time to evolve
        iznext = iz - 1                
        z = redshift[iz]
        tcurrent = CosmicTime[iz]
        tnext = CosmicTime[iznext]
        dt = tnext - tcurrent
        Dv = VirialOverdensity[iz]

        for level in levels: #loop from low-order to high-order systems
            for id in idx: # loop over branches
                if order[id,iz]!=level: continue # level by level
                if(iz <= izroot[id]):
                    if(iz == izroot[id]): # accretion happens at this timestep
                        # initialize Green profile and orbit

                        za = z
                        ta = tcurrent
                        Dva = Dv
                        ma = mass[id,iz] # initial mass that we will use for f_b
                        c2a = concentration[id,iz]
                        xva = coordinates[id,iz,:]

                        # some edge case produces nan in velocities in TreeGen
                        # if so, print warning and mass fraction lost
                        if(np.any(np.isnan(xva))):
                            print('    WARNING: NaNs detected in init xv of id %d'\
                                % id)
                            print('    Mass fraction of tree lost: %.1e'\
                                % (ma/mass[0,0]))
                            mass[id,:] = -99.
                            coordinates[id,:,:] = 0.
                            idx = np.delete(idx, np.argwhere(idx == id)[0])
                            # this is an extremely uncommon event, but should
                            # eventually be fixed
                            continue

                        potentials[id] = Green(ma,c2a,Delta=Dva,z=za)
                        orbits[id] = orbit(xva)
                        trelease[id] = ta

                        if cfg.evo_mode == 'arbres':
                            min_mass[id] = cfg.phi_res * ma
                        elif cfg.evo_mode == 'withering':
                            min_mass[id] = cfg.psi_res * M0

                    #---main loop for evolution

                    # the p,s,o objects are updated in-place in their arrays
                    # unless the orbit is replaced with a new object when released
                    ip = ParentID[id,iz]
                    p = potentials[ip]
                    s = potentials[id]

                    # update mass of subhalo object based on mass-loss in previous snapshot
                    # we wait to do it until now so that the pre-stripped subhalo can be used
                    # in the evolution of any higher-order subhaloes
                    # We also strip off the mass of any ejected systems
                    # the update_mass function handles cases where we fall below resolution limit
                    if(s.Mh > min_mass[id]):
                        if(ejected_mass[id] > 0):
                            mass[id,iz] -= ejected_mass[id]
                            ejected_mass[id] = 0
                            mass[id,iz] = max(mass[id,iz], cfg.phi_res*s.Minit)

                        s.update_mass(mass[id,iz])
                        rte = s.rte()

                    o = orbits[id]
                    xv = orbits[id].xv
                    m = s.Mh
                    m_old = m
                    r = np.sqrt(xv[0]**2+xv[2]**2)

                    #---time since in current host
                    t = tnext - trelease[id]

                    # Order should always be one higher than parent unless 
                    # ejected,in which case it should be the same as parent
                    k = order[ip,iznext] + 1

                    # alpha: stripping efficiency
                    if(alpha_type == 'fixed'):
                        alpha = 0.55
                    elif(alpha_type == 'conc'):
                        alpha = ev.alpha_from_c2(p.ch, s.ch)

                    #---evolve satellite
                    # as long as the mass is larger than resolution limit
                    if m > min_mass[id]:

                        # evolve subhalo properties
                        m,lt = ev.msub(s,p,xv,dt,choice='King62',
                            alpha=alpha)

                    else: # we do nothing about disrupted satellite, s.t.,
                        # its properties right before disruption would be 
                        # stored in the output arrays
                        pass

                    #---evolve orbit
                    if m > min_mass[id]:
                        # NOTE: We previously had an additional check on r>Rres
                        # here, where Rres = 10^-3 Rvir(z), but I removed it
                        # All subhalo orbits are evolved until their mass falls
                        # below the resolution limit.
                        # NOTE: No use integrating orbit any longer once the halo
                        # is disrupted, this just slows it down
                    
                        tdyn = p.tdyn(r)
                        o.integrate(t,p,m_old)
                        xv = o.xv # note that the coordinates are updated 
                        # internally in the orbit instance "o" when calling
                        # the ".integrate" method, here we assign them to 
                        # a new variable "xv" only for bookkeeping
                        
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
                                else:
                                    xv = aux.add_cyl_vecs(xv,coordinates[ip,iz,:])
                                    # This will be extraordinarily rare, but just
                                    # a check in case so that the released order-k
                                    # subhalo isn't accidentally double-released
                                    # in terms of updated coordinates, but not
                                    # in terms of new host ID.
                                orbits[id] = orbit(xv) # update orbit object
                                k = order[ip,iz] # update instant order to the same as the parent
                                ejected_mass[ip] += m 
                                # add updated subhalo mass to a bucket to be removed from host
                                # at start of next timestep
                                ip = ParentID[ip,iz] # update parent id
                                trelease[id] = tnext # update release time

                    #---update the arrays for output
                    mass[id,iznext] = m
                    order[id,iznext] = k
                    ParentID[id,iznext] = ip
                    try:
                        VirialRadius[id,iznext] = lt # storing tidal radius
                    except UnboundLocalError:
                        # TreeGen gives a few subhaloes with root mass below the
                        # given resolution limit so some subhaloes will never get
                        # an lt assigned if they aren't evolved one step. This can
                        # be fixed by lowering the resolution limit of SubEvo
                        # relative to TreeGen by some tiny epsilon, say 0.05 dex
                        print("No lt for id ", id, "iz ", iz, "masses ",
                              np.log10(mass[id,iz]), np.log10(mass[id,iznext]), file)
                        return

                    # NOTE: We store tidal radius in lieu of virial radius
                    # for haloes after they start getting stripped
                    GreenRte[id,iz] = rte 
                    coordinates[id,iznext,:] = xv

                    # NOTE: the below two are quantities at current timestep
                    # instead, since only used for host release criteria
                    # This won't be output since only used internally
                    alphas[id,iz] = alpha
                    tdyns[id,iz] = tdyn

                else: # before accretion, halo is an NFW profile
                    if(concentration[id,iz] > 0): 
                        # the halo has gone above tree mass resolution
                        # different than SatEvo mass resolution by small delta
                        potentials[id] = NFW(mass[id,iz],concentration[id,iz],
                                             Delta=VirialOverdensity[iz],z=redshift[iz])

    #---output
    np.savez(outfile, 
        redshift = redshift,
        CosmicTime = CosmicTime,
        mass = mass,
        order = order,
        ParentID = ParentID,
        VirialRadius = VirialRadius,
        GreenRte = GreenRte,
        # this contains values during stripping, -99 prior to stripping and
        # once the halo falls below the resolution limit
        concentration = concentration, # this is unchanged from TreeGen output
        coordinates = coordinates,
        )
    
    #---on-screen prints
    m0 = mass[:,0][1:]
    
    msk = (m0 > cfg.psi_res*M0) & (m0 < M0) & order[1:,0] == 1
    fsub = m0[msk].sum() / M0
    
    MAH = mass[0,:]
    iz50 = aux.FindNearestIndex(MAH,0.5*M0)
    z50 = redshift[iz50]
    
    time_end_tmp = time.time()
    print('    %s: %5.2f min, z50=%5.2f,fsub=%8.5f'%\
        (outfile,(time_end_tmp-time_start_tmp)/60., z50,fsub))
    sys.stdout.flush()

#---for parallelization, comment for testing in serial mode
if __name__ == "__main__":
    Ncores = int(sys.argv[1])
    pool = Pool(Ncores) # use as many as requested
    pool.map(loop, files, chunksize=1)

time_end = time.time() 
print('    total time: %5.2f hours'%((time_end - time_start)/3600.))
