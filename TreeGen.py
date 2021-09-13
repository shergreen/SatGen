################################ TreeGen ################################

# Generate halo merger trees using the Parkinson et al. (2008) algorithm.

# Arthur Fangzhou Jiang 2015 Yale University
# Arthur Fangzhou Jiang 2016 Hebrew University
# Arthur Fangzhou Jiang 2019 Hebrew University

######################## set up the environment #########################

#---user modules
import config as cfg
import cosmo as co
import init
from profiles import Dekel
import aux

#---python modules
import numpy as np
import time 
from multiprocessing import Pool, cpu_count
import sys

# <<< for clean on-screen prints, use with caution, make sure that 
# the warning is not prevalent or essential for the result
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

############################# user control ##############################

#---target halo, desired resolution, number of trees
lgM0_lo = 14.00
lgM0_hi = 14.50
z0 = 0.
lgMres = 8.5
Ntree = 24

#---baryonic-effect choice and output control
HaloResponse = 'NIHAO'
outfile1 = './OUTPUT_TREE_CLUSTER_NIHAO/tree%i_lgM%.2f.npz'#%(itree,lgM0)

#HaloResponse = 'APOSTLE'
#outfile1 = './OUTPUT_TREE_CLUSTER_APOSTLE/tree%i_lgM%.2f.npz'#%(itree,lgM0)

############################### compute #################################

print('>>> Generating %i trees for log(M_0)=%.2f-%.2f at log(M_res)=%.2f...'%\
    (Ntree,lgM0_lo,lgM0_hi,lgMres))

#---
time_start = time.time()
#for itree in range(Ntree):
def loop(itree): 
    """
    Replaces the loop "for itree in range(Ntree):", for parallelization.
    """
    time_start_tmp = time.time()
    
    np.random.seed() # [important!] reseed the random number generator
    
    lgM0 = lgM0_lo + np.random.random()*(lgM0_hi-lgM0_lo)
    cfg.M0 = 10.**lgM0
    cfg.z0 = z0
    cfg.Mres = 10.**lgMres 
    cfg.Mmin = 0.04*cfg.Mres
    
    k = 0               # the level, k, of the branch being considered
    ik = 0              # how many level-k branches have been finished
    Nk = 1              # total number of level-k branches
    Nbranch = 1         # total number of branches in the current tree
    
    Mak = [cfg.M0]      # accretion masses of level-k branches
    zak = [cfg.z0]
    idk = [0]           # branch ids of level-k branches
    ipk = [-1]          # parent ids of level-k branches (-1: no parent) 
    
    Mak_tmp = []
    zak_tmp = []
    idk_tmp = []
    ipk_tmp = []
    
    mass = np.zeros((cfg.Nmax,cfg.Nz)) - 99.
    order = np.zeros((cfg.Nmax,cfg.Nz),np.int8) - 99
    ParentID = np.zeros((cfg.Nmax,cfg.Nz),np.int16) - 99
    
    VirialRadius = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    concentration = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    DekelConcentration = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    DekelSlope = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99.
    
    StellarMass = np.zeros((cfg.Nmax,cfg.Nz)) - 99. 
    StellarSize = np.zeros((cfg.Nmax,cfg.Nz),np.float32) - 99. 
    
    coordinates = np.zeros((cfg.Nmax,cfg.Nz,6),np.float32)
    
    while True: # loop over branches, until the full tree is completed.
    # Starting from the main branch, draw progenitor(s) using the 
    # Parkinson+08 algorithm. When there are two progenitors, the less 
    # massive one is the root of a new branch. We draw branches level by
    # level, i.e., When a new branch occurs, we record its root, but keep 
    # finishing the current branch and all the branches of the same level
    # as the current branch, before moving on to the next-level branches.
    
        M = [Mak[ik]]   # mass history of current branch in fine timestep
        z = [zak[ik]]   # the redshifts of the mass history
        cfg.M0 = Mak[ik]# descendent mass
        cfg.z0 = zak[ik]# descendent redshift
        id = idk[ik]    # branch id
        ip = ipk[ik]    # parent id
        
        while cfg.M0>cfg.Mmin:
        
            if cfg.M0>cfg.Mres: zleaf = cfg.z0 # update leaf redshift
        
            co.UpdateGlobalVariables(**cfg.cosmo)
            M1,M2,Np = co.DrawProgenitors(**cfg.cosmo)
            
            # update descendent halo mass and descendent redshift 
            cfg.M0 = M1
            cfg.z0 = cfg.zW_interp(cfg.W0+cfg.dW)
            if cfg.z0>cfg.zmax: break
            
            if Np>1 and cfg.M0>cfg.Mres: # register next-level branches
            
                Nbranch += 1
                Mak_tmp.append(M2)
                zak_tmp.append(cfg.z0)
                idk_tmp.append(Nbranch)
                ipk_tmp.append(id)
            
            # record the mass history at the original time resolution
            M.append(cfg.M0)
            z.append(cfg.z0)
        
        # Now that a branch is fully grown, do some book-keeping
        
        # convert mass-history list to array 
        M = np.array(M)
        z = np.array(z)
        
        # downsample the fine-step mass history, M(z), onto the
        # coarser output timesteps, cfg.zsample     
        Msample,zsample = aux.downsample(M,z,cfg.zsample)
        iz = aux.FindClosestIndices(cfg.zsample,zsample)
        izleaf = aux.FindNearestIndex(cfg.zsample,zleaf)
        
        # compute halo structure throughout time on the coarse grid, up
        # to the leaf point
        t = co.t(z,cfg.h,cfg.Om,cfg.OL)
        c,a,c2,Rv = [],[],[],[]
        for i in iz:
            if i > (izleaf+1): break # only compute structure below leaf
            msk = z>=cfg.zsample[i]
            if True not in msk: break # safety
            ci,ai,Msi,c2i,c2DMOi = init.Dekel_fromMAH(M[msk],t[msk],
                cfg.zsample[i],HaloResponse=HaloResponse)
            Rvi = init.Rvir(M[msk][0],Delta=200.,z=cfg.zsample[i])
            c.append(ci)
            a.append(ai)
            c2.append(c2i)
            Rv.append(Rvi)
            if i==iz[0]: Ms = Msi
            #print('    i=%6i,ci=%8.2f,ai=%8.2f,log(Msi)=%8.2f,c2i=%8.2f'%\
            #    (i,ci,ai,np.log10(Msi),c2i)) # <<< for test
        if len(c)==0: # <<< safety, dealing with rare cases where the 
            # branch's root z[0] is close to the maximum redshift -- when
            # this happens, the mass history has only one element, and 
            # z[0] can be slightly above cfg.zsample[i] for the very 
            # first iteration, leaving the lists c,a,c2,Rv never updated
            ci,ai,Msi,c2i=init.Dekel_fromMAH(M,t,z[0],
                HaloResponse=HaloResponse)
            c.append(ci)
            a.append(ai)
            c2.append(c2i)
            Rv.append(Rvi)
            Ms = Msi
            # <<< test
            #print('    branch root near max redshift: z=%7.2f,log(M)=%7.2f,c=%7.2f,a=%7.2f,c2=%7.2f,log(Ms)=%7.2f'%\
            #    (z[0],np.log10(M[0]),c[0],a[0],c2[0],np.log10(Ms))) 
        c = np.array(c)
        a = np.array(a) 
        c2 = np.array(c2) 
        Rv = np.array(Rv)
        Nc = len(c2) # length of a branch over which c2 is computed 
        
        # compute stellar size at the root of the branch, i.e., at the 
        # accretion epoch (z[0])
        Re = init.Reff(Rv[0],c2[0])
        
        # use the redshift id and parent-branch id to access the parent
        # branch's information at our current branch's accretion epoch,
        # in order to initialize the orbit
        if ip==-1: # i.e., if the branch is the main branch
            xv = np.zeros(6)
        else:
            Mp = mass[ip,iz[0]]
            cp = DekelConcentration[ip,iz[0]]
            ap = DekelSlope[ip,iz[0]]
            hp = Dekel(Mp,cp,ap,Delta=200.,z=zsample[0])
            eps = 1./np.pi*np.arccos(1.-2.*np.random.random())
            xv = init.orbit(hp,xc=1.,eps=eps)
        
        # <<< test
        #print('    id=%6i,k=%2i,z=%7.2f,log(M)=%7.2f,c=%7.2f,a=%7.2f,c2=%7.2f,log(Ms)=%7.2f,Re=%7.2f,xv=%7.2f,%7.2f,%7.2f,%7.2f,%7.2f,%7.2f'%\
        #    (id,k,z[0],np.log10(M[0]),c[0],a[0],c2[0],np.log10(Ms),Re, xv[0],xv[1],xv[2],xv[3],xv[4],xv[5]))
        
        # update the arrays for output
        mass[id,iz] = Msample
        order[id,iz] = k
        ParentID[id,iz] = ip
        
        VirialRadius[id,iz[0]:iz[0]+Nc] = Rv
        concentration[id,iz[0]:iz[0]+Nc] = c2
        DekelConcentration[id,iz[0]:iz[0]+Nc] = c
        DekelSlope[id,iz[0]:iz[0]+Nc] = a

        StellarMass[id,iz[0]] = Ms
        StellarSize[id,iz[0]] = Re
        
        coordinates[id,iz[0],:] = xv
                
        # Check if all the level-k branches have been dealt with: if so, 
        # i.e., if ik==Nk, proceed to the next level.
        ik += 1
        if ik==Nk: # all level-k branches are done!
            Mak = Mak_tmp
            zak = zak_tmp
            idk = idk_tmp
            ipk = ipk_tmp
            Nk = len(Mak)
            ik = 0
            Mak_tmp = []
            zak_tmp = []
            idk_tmp = []
            ipk_tmp = []
            if Nk==0: 
                break # jump out of "while True" if no next-level branch 
            k += 1 # update level
    
    # trim and output 
    mass = mass[:id+1,:]
    order = order[:id+1,:]
    ParentID = ParentID[:id+1,:]
    VirialRadius = VirialRadius[:id+1,:]
    concentration = concentration[:id+1,:]
    DekelConcentration = DekelConcentration[:id+1,:]
    DekelSlope = DekelSlope[:id+1,:]
    StellarMass = StellarMass[:id+1,:]
    StellarSize = StellarSize[:id+1,:]
    coordinates = coordinates[:id+1,:,:]
    np.savez(outfile1%(itree,lgM0), 
        redshift = cfg.zsample,
        CosmicTime = cfg.tsample,
        mass = mass,
        order = order,
        ParentID = ParentID,
        VirialRadius = VirialRadius,
        concentration = concentration,
        DekelConcentration = DekelConcentration,
        DekelSlope = DekelSlope,
        #VirialOverdensity = VirialOverdensity, # <<< no need in TreeGen
        StellarMass = StellarMass,
        StellarSize = StellarSize,
        coordinates = coordinates,
        )
            
    time_end_tmp = time.time()
    print('    Tree %5i: log(M_0)=%6.2f, %6i branches, %2i order, %8.1f sec'\
        %(itree,lgM0,Nbranch,k,time_end_tmp-time_start_tmp))

#---for parallelization, comment for testing in serial mode
if __name__ == "__main__":
    pool = Pool(cpu_count()) # use all cores
    pool.map(loop, range(Ntree))

time_end = time.time() 
print('    total time: %5.2f hours'%((time_end - time_start)/3600.))
