#########################################################################

# Program that test the modules for mass stripping and orbit integration 
# (evolve.py and orbit.py) in potential wells defined in profiles.py   

# Arthur Fangzhou Jiang 2019 HUJI

######################## set up the environment #########################

#---user modules
import config as cfg
import cosmo as co
from profiles import NFW,Dekel,Einasto,MN, Vcirc,ftot,fDF,tdyn
from orbit import orbit
import evolve as ev
import galhalo as gh
import aux

#---python modules
import numpy as np
import time 
import sys

#---for plot
import matplotlib as mpl # must import before pyplot
mpl.use('Qt5Agg')
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 15
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

########################### user control ################################

#---host properties <<< use the same as used in test_profiles.py
Mv = 10.**12. # halo virial mass [Msun]
cNFW = 10. # NFW concentration
cDekel = 10. # Dekel+ concentration
aDekel = cfg.eps + 0.53
cEinasto = 10. # Einasto concentration
aEinasto = 0.18
fgh = 0.1 # halo (hot) gas fraction in units of halo mass
#
Md = 5e10
aMN = 5. # Miyamoto-Nagai disk scale radius
bMN = 1. # Miyamoto-Nagai disk scale height
fgd = 1e-6 # disk (cold) gas fraction in units of disk mass
#
hNFW = NFW(Mv,cNFW)
hDekel = Dekel(Mv,cDekel,aDekel)
gDekel = Dekel(Mv*fgh,cDekel,aDekel,Delta=hDekel.Deltah*fgh) # gas halo 
hEinasto = Einasto(Mv,cEinasto,aEinasto)
dMN = MN(Md,aMN,bMN)
gMN = MN(Md*fgd,aMN,bMN) # gas disk

#---choose the potential to study
#potential = dMN
potential = hDekel
potential = [dMN,hDekel]
#gpotential = gDekel
gpotential = [gMN,gDekel]

#---satellite properties <<< play with
mv0 = 10.**11.#0.1 * Mv # satellite virial mass at infall [Msun]
cDekel0 = 20. # initial concentration of satellite
aDekel0 = cfg.eps + 0.
fg0 = 0.1 # initial gas fraction <<< play with
ms0 = 10.**9. #0.01 * mv0 # initial stellar mass <<< play with
#
s = Dekel(mv0,cDekel0,aDekel0)
sg = Dekel(mv0*fg0,cDekel0,aDekel0,Delta=s.Deltah*fg0)
# <<< test
s0 = s
sg0 =sg

#---initial orbit control <<< play with
R0 = 55.
#R0 = 0.8*hDekel.rh
phi0 = 0.
z0 = 30.
#z0 = 0.6*hDekel.rh
VR0 = 0.
Vphi0 = Vcirc(potential,np.sqrt(R0**2+z0**2),0.)
Vz0 = 0.
#
xv0 = np.array([R0,phi0,z0,VR0,Vphi0,Vz0])

#---for evolution and bookkeeping
Nstep = 200 # number of timesteps
tmax = 5. # [Gyr] 
timesteps = np.linspace(0.,tmax,Nstep+1)[1::] #[Gyr]

#---for plotting satellite profiles
llv0_grid = 10.**np.linspace(-2.5,0.3,100)
ttdyn0_grid = np.array([0.,0.5,1.,1.5,2.,2.5])
tdyn0 = tdyn(potential,np.sqrt(R0**2+z0**2)) # initial orbital time
t_grid = ttdyn0_grid * tdyn0
#t_grid = np.array([0.,0.5,1.,1.5,2.,2.5,3.,3.5])

#---choosing the "final" time for deciding if the satellite penetrates
tfinaltdyn0 = 3. # in units of the initial dynamical time  
r_min = 0.02*hDekel.rh # radius [kpc] at which we register the final time

#---plot control
Rmax = 1.1*np.sqrt(R0**2+z0**2)
lw=2
outfig1 = './FIGURE/test_evolve.pdf'

def fcolor(v,vmin=0.,vmax=t_grid.max(),choice='t'):
    r"""
    Returns 
    
        - a color,  
        - the color map from which the color is drawn,
        - the normalization of the color map, 
    
    upon input of a property.
    
    Syntax: 
    
        fcolor(v,vmin=vmin,vmax=vmax,choice='...')
    
    where 
    
        v: the value of the halo property of choice
        vmin: the value below which there is no color difference
        vmax: the value above which there is no color difference
        choice: ...
    """
    if choice == 't': 
        cm = plt.cm.coolwarm
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        return scalarMap.to_rgba(v),cm,norm
        
############################### compute #################################

print('>>> initializing ... ')
o = orbit(xv0)
r = np.sqrt(R0**2+z0**2)
m = mv0
lmax0 = s.rmax
vmax0 = s.Vcirc(lmax0)
lv0 = s.rh 
leff0 = gh.Reff(lv0,s.rh/s.r2)
lefflmax0 = leff0 / lmax0
mmax0 = s.M(lmax0)

radius = np.zeros(Nstep)
velocity = np.zeros(Nstep)
xpos = np.zeros(Nstep)
ypos = np.zeros(Nstep)
zpos = np.zeros(Nstep)
mass = np.zeros(Nstep)
GasMass = np.zeros(Nstep)
StellarMass = np.zeros(Nstep)
TidalRadius = np.zeros(Nstep)
RamPressureRadius = np.zeros(Nstep)
EffectiveRadius = np.zeros(Nstep)
MassLossRate = np.zeros(Nstep)
GasLossRate = np.zeros(Nstep)
concentration = np.zeros(Nstep)
slope = np.zeros(Nstep)

l_grid = lv0 * llv0_grid 
rho_grid = np.zeros((l_grid.size,t_grid.size))
vc_grid = np.zeros((l_grid.size,t_grid.size))
rhog_grid = np.zeros((l_grid.size,t_grid.size))
vcg_grid = np.zeros((l_grid.size,t_grid.size))
rho_grid[:,0] = s.rho(l_grid)
vc_grid[:,0] = s.Vcirc(l_grid)
rhog_grid[:,0] = sg.rho(l_grid)
vcg_grid[:,0] = sg.Vcirc(l_grid)

tprevious = 0.

print('>>> evolving ... ')
t1 = time.time()
for i,t in enumerate(timesteps):
    
    dt = t -  tprevious
    
    #---evolve orbit
    if r>cfg.Rres:
        o.integrate(t,potential,m)
        xv = o.xv
        # note that the coordinates are updated internally in the orbit 
        # instance "o", here we assign them to xv only for bookkeeping
    else: 
        continue
    r = np.sqrt(xv[0]**2+xv[2]**2)
    V = np.sqrt(xv[3]**2+xv[4]**2+xv[5]**2)
    x = xv[0]*np.cos(xv[1])
    y = xv[0]*np.sin(xv[1])
    z = xv[2]
    
    #---evolve subhalo mass and profile
    if m>cfg.Mres:
        m,lt = ev.msub(s,potential,xv,dt,choice='King62',alpha=1.)
        a = s.alphah # assume: innermost slope not affected by tides 
        c,Delta = ev.Dekel(m,mv0,lmax0,vmax0,aDekel0,z=0.)
    else:
        m = cfg.Mres
        lt = cfg.Rres
        c = s.ch
        a = s.alphah
    dmdt = (s.Mh - m) / dt 
    s = Dekel(m,c,a,Delta=Delta,z=0.)
    
    #---evolve satellite gas and profile
    if sg.Mh>cfg.Mres:
        mg,lrp = ev.mgas(sg,s,gpotential,potential,xv,dt,kappa=1.,
            alpha=1.)
    else:
        mg = cfg.Mres
        lrp = cfg.Rres
    dmgdt = (sg.Mh - mg) / dt 
    sg = Dekel(mg,c,a,Delta=Delta*(mg/m))
    
    #---evolve satellite stellar size
    mmax = s.M(s.rmax) 
    g_leff, g_mstar = ev.g_EPW18(mmax/mmax0,aDekel0,lefflmax0)
    leff = leff0 * g_leff
    ms = ms0 * g_mstar 
    ms = min(ms,m) # <<< safety
    
    #---record
    radius[i] = r
    velocity[i] = V
    mass[i] = m
    StellarMass[i] = ms
    GasMass[i] = mg
    xpos[i] = x
    ypos[i] = y
    zpos[i] = z
    TidalRadius[i] = lt
    RamPressureRadius[i] = lrp
    EffectiveRadius[i] = leff
    MassLossRate[i] = dmdt
    GasLossRate[i] = dmgdt
    concentration[i] = s.rh / s.rmax
    slope[i] = s.sh
    
    #---record profile at sampling times
    if tprevious<t_grid[1] and t>=t_grid[1]:
        rho_grid[:,1] = s.rho(l_grid)
        vc_grid[:,1] = s.Vcirc(l_grid)
        rhog_grid[:,1] = sg.rho(l_grid)
        vcg_grid[:,1] = sg.Vcirc(l_grid)
    elif tprevious<t_grid[2] and t>=t_grid[2]:
        rho_grid[:,2] = s.rho(l_grid)
        vc_grid[:,2] = s.Vcirc(l_grid)
        rhog_grid[:,2] = sg.rho(l_grid)
        vcg_grid[:,2] = sg.Vcirc(l_grid)
    elif tprevious<t_grid[3] and t>=t_grid[3]:
        rho_grid[:,3] = s.rho(l_grid)
        vc_grid[:,3] = s.Vcirc(l_grid)
        rhog_grid[:,3] = sg.rho(l_grid)
        vcg_grid[:,3] = sg.Vcirc(l_grid)
    elif tprevious<t_grid[4] and t>=t_grid[4]:
        rho_grid[:,4] = s.rho(l_grid)
        vc_grid[:,4] = s.Vcirc(l_grid)
        rhog_grid[:,4] = sg.rho(l_grid)
        vcg_grid[:,4] = sg.Vcirc(l_grid)
    elif tprevious<t_grid[5] and t>=t_grid[5]:
        rho_grid[:,5] = s.rho(l_grid)
        vc_grid[:,5] = s.Vcirc(l_grid)
        rhog_grid[:,5] = sg.rho(l_grid)
        vcg_grid[:,5] = sg.Vcirc(l_grid)
    
    #---update tprevious
    tprevious = t

t2 = time.time()
print('    time = %.4f'%(t2-t1))

#---a few other quantities for plot
Vv = Vcirc(potential,hDekel.rh) # host halo virial velocity
#if radius.min() < r_min:
#    idx = aux.FindNearestIndex(radius,r_min)
#    tfinal = timesteps[idx]
#else:
#    tfinal = tfinaltdyn0*tdyn0
#    idx = aux.FindNearestIndex(timesteps,tfinal)
tfinal = tfinaltdyn0*tdyn0 # <<< test  
idx = aux.FindNearestIndex(timesteps,tfinal) # <<< test
mfinal = mass[idx]
print('    m(t_final)/m(0)=%.4f'%(mfinal/mv0))

########################### diagnostic plots ############################

print('>>> plot ...')
plt.close('all') # close all previous figure windows

#------------------------------------------------------------------------

# set up the figure window
fig1 = plt.figure(figsize=(16,9), dpi=80, facecolor='w', edgecolor='k') 
fig1.subplots_adjust(left=0.06, right=0.95,bottom=0.06, top=0.99,
    hspace=0.25, wspace=0.47)
gs = gridspec.GridSpec(3, 5) 
fig1.suptitle(r'')

#---
ax = fig1.add_subplot(gs[1:3,1:3],projection='3d')
#ax.view_init(elev=45., azim=-60) # manually adjust viewing angle
ax.set_xlim(-R0,R0)
ax.set_ylim(-R0,R0)
ax.set_zlim(-R0,R0) 
ax.set_xlabel(r'$x$ [kpc]',fontsize=16)
ax.set_ylabel(r'$y$ [kpc]',fontsize=16)
ax.set_zlabel(r'$z$ [kpc]',fontsize=16)
# tick label size
ax.xaxis.set_tick_params(labelsize=12)
ax.yaxis.set_tick_params(labelsize=12)
ax.zaxis.set_tick_params(labelsize=12)
#ax.set_title(r'')
# plot
ax.plot(xpos,ypos,zpos,lw=lw,color='k',)
# annotation
ax.text2D(-0.1,0.95,r'a)',
    ha='left',va='center',transform=ax.transAxes,rotation=0)
#
ax.text2D(0.,0.95,r'$t=%.2f$ Gyr'%tmax,fontsize=16,
    ha='left',va='center',transform=ax.transAxes,rotation=0)
#
ax.text2D(-0.09,0.7,r'$R_0=%.1f$ kpc'%R0,fontsize=16,
    ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(-0.09,0.65,r'$\phi_0=%.2f$'%phi0,fontsize=16,
    ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(-0.09,0.6,r'$z_0=%.1f$ kpc'%z0,fontsize=16,
    ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(-0.09,0.55,r'$V_{R,0}=%.1f$ kpc/Gyr'%VR0,fontsize=16,
    ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(-0.09,0.5,r'$V_{\phi,0}=%.1f$ kpc/Gyr'%Vphi0,fontsize=16,
    ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(-0.09,0.45,r'$V_{z,0}=%.1f$ kpc/Gyr'%Vz0,fontsize=16,
    ha='left',va='center',transform=ax.transAxes,rotation=0)
#
ax.text2D(0.35,0.95,r'$M_\mathrm{v}=10^{%.1f}M_\odot$'%np.log10(Mv),
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(0.35,0.9,r'$c=%.1f$'%cDekel,
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(0.65,0.9,r'$\alpha=%.1f$'%aDekel,
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
#
ax.text2D(0.35,0.85,r'$M_\mathrm{d}=10^{%.1f}M_\odot$'%np.log10(Md),
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(0.35,0.8,r'$a=%.1f$ kpc'%aMN,
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(0.65,0.8,r'$b=%.1f$ kpc'%bMN,
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
#
ax.text2D(0.4,0.25,r'$m_\mathrm{v,0}=10^{%.1f}M_\odot$'%np.log10(mv0),
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(0.4,0.2,r'$c_0=%.1f$'%cDekel0,
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
ax.text2D(0.4,0.15,r'$\alpha_0=%.1f$'%aDekel0,
    fontsize=18,ha='left',va='center',transform=ax.transAxes,rotation=0)
# legends
#ax.legend(loc='upper right',numpoints=1,scatterpoints=1,fontsize=16,
#    frameon=True) #bbox_to_anchor=(1.2, 0.9)

#---
ax = fig1.add_subplot(gs[0,0])
ax.set_xlim(-R0,R0)
ax.set_ylim(-R0,R0)
ax.set_xlabel(r'$x$ [kpc]')
ax.set_ylabel(r'$y$ [kpc]')
if (Rmax >= 100.):
    start = - 100.
    end = -start
    major_ticks = np.arange(start, end, 50.)
    minor_ticks = np.arange(start, end, 10.)
elif (Rmax >= 50.):
    start = - 50.
    end = -start
    major_ticks = np.arange(start, end, 25.)
    minor_ticks = np.arange(start, end, 5.)
elif (Rmax >= 10.):
    start = - 10.
    end = -start
    major_ticks = np.arange(start, end, 5.)
    minor_ticks = np.arange(start, end, 1.)
elif (Rmax >= 2.):
    start = - 2.
    end = -start
    major_ticks = np.arange(start, end, 2.)
    minor_ticks = np.arange(start, end, 0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
# plot
ax.plot(xpos,ypos,lw=lw,color='k')
# annotation
ax.text(0.05,0.87,r'b)',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transAxes)

#---
ax = fig1.add_subplot(gs[0,1])
ax.set_xlim(-R0,R0)
ax.set_ylim(-R0,R0)
ax.set_xlabel(r'$z$ [kpc]')
ax.set_ylabel(r'$y$ [kpc]')
#ax.yaxis.set_major_formatter(plt.NullFormatter()) # hide y tick labels
if (Rmax >= 100.):
    start = - 100.
    end = -start
    major_ticks = np.arange(start, end, 50.)
    minor_ticks = np.arange(start, end, 10.)
elif (Rmax >= 50.):
    start = - 50.
    end = -start
    major_ticks = np.arange(start, end, 25.)
    minor_ticks = np.arange(start, end, 5.)
elif (Rmax >= 10.):
    start = - 10.
    end = -start
    major_ticks = np.arange(start, end, 5.)
    minor_ticks = np.arange(start, end, 1.)
elif (Rmax >= 2.):
    start = - 2.
    end = -start
    major_ticks = np.arange(start, end, 2.)
    minor_ticks = np.arange(start, end, 0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major',)
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
# plot
ax.plot(zpos,ypos,lw=lw,color='k')
# annotation
ax.text(0.05,0.87,r'c)',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transAxes)

#---
ax = fig1.add_subplot(gs[1,0])
ax.set_xlim(-R0,R0)
ax.set_ylim(-R0,R0)
ax.set_xlabel(r'$x$ [kpc]')
ax.set_ylabel(r'$z$ [kpc]')
if (Rmax >= 100.):
    start = - 100.
    end = -start
    major_ticks = np.arange(start, end, 50.)
    minor_ticks = np.arange(start, end, 10.)
elif (Rmax >= 50.):
    start = - 50.
    end = -start
    major_ticks = np.arange(start, end, 25.)
    minor_ticks = np.arange(start, end, 5.)
elif (Rmax >= 10.):
    start = - 10.
    end = -start
    major_ticks = np.arange(start, end, 5.)
    minor_ticks = np.arange(start, end, 1.)
elif (Rmax >= 2.):
    start = - 2.
    end = -start
    major_ticks = np.arange(start, end, 2.)
    minor_ticks = np.arange(start, end, 0.5)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
# plot
ax.plot(xpos,zpos,lw=lw,color='k')
# annotation
ax.text(0.05,0.87,r'd)',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transAxes)

#---
ax = fig1.add_subplot(gs[0,2])
ax.set_xlim(0.01*lv0,2.*lv0)
ax.set_ylim((rho_grid[:,0]*l_grid**2).max()*0.005,
    (rho_grid[:,0]*l_grid**2).max()*2)
ax.set_xlabel(r'$l$ [kpc]')
#ax.set_ylabel(r'$\rho$ [$M_\odot$ kpc$^{-3}$]')
ax.set_ylabel(r'$l^2\rho(l)$ [$M_\odot$ kpc$^{-1}$]')
#ax.set_title(r'')
# scale
ax.set_xscale('log')
ax.set_yscale('log')
# tick and tick label positions
#start,end = ax.get_xlim()
#major_ticks = np.arange(start, end, 10.)
#minor_ticks = np.arange(start, end, 2.)
#ax.set_xticks(major_ticks)
#ax.set_xticks(minor_ticks,minor=True)
# for refined control of log-scale tick marks
locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
locmin = mpl.ticker.LogLocator(base=10.0,
    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
#
locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
locmin = mpl.ticker.LogLocator(base=10.0,
    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    numticks=12)
ax.yaxis.set_major_locator(locmaj)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# grid
#ax.grid(which='minor', alpha=0.2)                                                
#ax.grid(which='major', alpha=0.4)
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
# plot
ax.plot(l_grid,rho_grid[:,0]*l_grid**2,lw=lw,color=fcolor(t_grid[0])[0])
ax.plot(l_grid,rho_grid[:,1]*l_grid**2,lw=lw,color=fcolor(t_grid[1])[0])
ax.plot(l_grid,rho_grid[:,2]*l_grid**2,lw=lw,color=fcolor(t_grid[2])[0])
ax.plot(l_grid,rho_grid[:,3]*l_grid**2,lw=lw,color=fcolor(t_grid[3])[0])
ax.plot(l_grid,rho_grid[:,4]*l_grid**2,lw=lw,color=fcolor(t_grid[4])[0])
ax.plot(l_grid,rho_grid[:,5]*l_grid**2,lw=lw,color=fcolor(t_grid[5])[0])
#
#ax.plot(l_grid,rhog_grid[:,0]*l_grid**2,lw=lw,ls='--',
#    color=fcolor(t_grid[0])[0])
#ax.plot(l_grid,rhog_grid[:,1]*l_grid**2,lw=lw,ls='--',
#    color=fcolor(t_grid[1])[0])
#ax.plot(l_grid,rhog_grid[:,2]*l_grid**2,lw=lw,ls='--',
#    color=fcolor(t_grid[2])[0])
#ax.plot(l_grid,rhog_grid[:,3]*l_grid**2,lw=lw,ls='--',
#    color=fcolor(t_grid[3])[0])
#ax.plot(l_grid,rhog_grid[:,4]*l_grid**2,lw=lw,ls='--',
#    color=fcolor(t_grid[4])[0])
#ax.plot(l_grid,rhog_grid[:,5]*l_grid**2,lw=lw,ls='--',
#    color=fcolor(t_grid[5])[0])
#
# reference line
ax.plot(np.repeat(lv0,2),ax.get_ylim(),ls=':',color='k')
#ax.plot(np.repeat(lmax0,2),ax.get_ylim(),ls='-.',color='k')
# annotations
ax.text(lv0,ax.get_ylim()[1]*0.2,r'$l_\mathrm{v}(0)$',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transData,rotation=90)
#
ax.text(0.2,0.45,r'$t=%.2f$Gyr'%t_grid[0],color=fcolor(t_grid[0])[0],
    fontsize=16,ha='left',va='bottom',transform=ax.transAxes,rotation=0)
ax.text(0.2,0.38,r'$t=%.2f$'%t_grid[1],color=fcolor(t_grid[1])[0],
    fontsize=16,ha='left',va='bottom',transform=ax.transAxes,rotation=0)
ax.text(0.2,0.31,r'$t=%.2f$'%t_grid[2],color=fcolor(t_grid[2])[0],
    fontsize=16,ha='left',va='bottom',transform=ax.transAxes,rotation=0)
ax.text(0.2,0.24,r'$t=%.2f$'%t_grid[3],color=fcolor(t_grid[3])[0],
    fontsize=16,ha='left',va='bottom',transform=ax.transAxes,rotation=0)
ax.text(0.2,0.16,r'$t=%.2f$'%t_grid[4],color=fcolor(t_grid[4])[0],
    fontsize=16,ha='left',va='bottom',transform=ax.transAxes,rotation=0)
ax.text(0.2,0.09,r'$t=%.2f$'%t_grid[5],color=fcolor(t_grid[5])[0],
    fontsize=16,ha='left',va='bottom',transform=ax.transAxes,rotation=0)
# legends
#ax.legend(loc='upper right',numpoints=1,scatterpoints=1,fontsize=14,
#    frameon=True) #bbox_to_anchor=(1.2, 0.9)
# annotation
ax.text(0.05,0.87,r'e)',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transAxes)
    
#---
ax = fig1.add_subplot(gs[2,0])
ax.set_xlim(0.01*lv0,2.*lv0)
ax.set_ylim(0.,vmax0*1.2)
ax.set_xlabel(r'$l$ [kpc]')
ax.set_ylabel(r'$v_\mathrm{circ}(l)$ [kpc/Gyr]')
#ax.set_title(r'')
# scale
ax.set_xscale('log')
#ax.set_yscale('log')
# tick and tick label positions
start,end = ax.get_ylim()
major_ticks = np.arange(start, end, 50.)
minor_ticks = np.arange(start, end, 10.)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
# for refined control of log-scale tick marks
locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
locmin = mpl.ticker.LogLocator(base=10.0,
    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# grid
#ax.grid(which='minor', alpha=0.2)                                                
#ax.grid(which='major', alpha=0.4)
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
# plot
ax.plot(l_grid,vc_grid[:,0],lw=lw,color=fcolor(t_grid[0])[0])
ax.plot(l_grid,vc_grid[:,1],lw=lw,color=fcolor(t_grid[1])[0])
ax.plot(l_grid,vc_grid[:,2],lw=lw,color=fcolor(t_grid[2])[0])
ax.plot(l_grid,vc_grid[:,3],lw=lw,color=fcolor(t_grid[3])[0])
ax.plot(l_grid,vc_grid[:,4],lw=lw,color=fcolor(t_grid[4])[0])
ax.plot(l_grid,vc_grid[:,5],lw=lw,color=fcolor(t_grid[5])[0])
#
#ax.plot(l_grid,vcg_grid[:,0],lw=lw,ls='--',color=fcolor(t_grid[0])[0])
#ax.plot(l_grid,vcg_grid[:,1],lw=lw,ls='--',color=fcolor(t_grid[1])[0])
#ax.plot(l_grid,vcg_grid[:,2],lw=lw,ls='--',color=fcolor(t_grid[2])[0])
#ax.plot(l_grid,vcg_grid[:,3],lw=lw,ls='--',color=fcolor(t_grid[3])[0])
#ax.plot(l_grid,vcg_grid[:,4],lw=lw,ls='--',color=fcolor(t_grid[4])[0])
#ax.plot(l_grid,vcg_grid[:,5],lw=lw,ls='--',color=fcolor(t_grid[5])[0])
# reference line
ax.plot(np.repeat(lv0,2),ax.get_ylim(),ls=':',color='k')
# annotations
ax.text(lv0,ax.get_ylim()[1]*0.8,r'$l_\mathrm{v}(0)$',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transData,rotation=90)
# legends
#ax.legend(loc='upper left',numpoints=1,scatterpoints=1,fontsize=16,
#    frameon=True) #bbox_to_anchor=(1.2, 0.9)
# annotation
ax.text(0.05,0.87,r'f)',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transAxes)
    
#---
ax = fig1.add_subplot(gs[0,3:])
ax.set_xlim(0.,3.)
ax.set_ylim(0.,Rmax)
ax.set_xlabel(r'$t$ [Gyr]')
ax.set_ylabel(r'$r$ [kpc]')
#ax.set_title(r'')
# tick and tick label positions
if tmax >=2.5: 
    start, end = 0., (tmax//2.5)*2.5
    major_ticks = np.arange(start, end, 2.)
    minor_ticks = np.arange(start, end, 0.5)
else:
    start, end = 0., (tmax//0.5)*0.5
    major_ticks = np.arange(start, end, 0.5)
    minor_ticks = np.arange(start, end, 0.1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
start = 0.
if (Rmax >= 100.):
    end = (Rmax//100.+1) *100.
    major_ticks = np.arange(start, end, 50.)
    minor_ticks = np.arange(start, end, 10.)
elif (Rmax >= 50.) and (Rmax<100.):
    end = (Rmax//20.+1) *20.
    major_ticks = np.arange(start, end, 25.)
    minor_ticks = np.arange(start, end, 5.)
else:
    end = (Rmax//10.+1) *10.
    major_ticks = np.arange(start, end, 10.)
    minor_ticks = np.arange(start, end, 2.)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
# scale
#ax.set_xscale('log')
#ax.set_yscale('log')
# grid
#ax.grid(which='minor', alpha=0.2)                                                
#ax.grid(which='major', alpha=0.4)
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
#
ax2 = ax.twinx()
ax2.set_ylabel(r'$V$ [km/s]',color='crimson')
ax2.set_ylim(0.,Vv*3)
start = 0.
end = (Vv*3//100.) *100.
major_ticks = np.arange(start, end, 50.)
minor_ticks = np.arange(start, end, 10.)
ax2.set_yticks(major_ticks)
ax2.set_yticks(minor_ticks,minor=True)
#ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.tick_params('y',colors='crimson',direction='in',
    right='on',length=10,width=1,which='major')
ax2.tick_params('y',colors='crimson',direction='in',
    right='on',length=5,width=1,which='minor')
#
# plot
ax.plot(timesteps,radius,lw=lw,color='k')
#ax.plot(timesteps,np.repeat(rperi,Nstep),lw=lw,color='grey',ls='--')
#ax.plot(timesteps,np.repeat(rapo,Nstep),lw=lw,color='grey',ls='--')
#
ax2.plot(timesteps,velocity,lw=lw,color='crimson')
# reference line
#ax.plot(np.repeat(tfinal,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(2.*tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(3.*tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')

# annotations
#ax.text(ax.get_xlim()[1],rperi,r'peri-center',color='k',fontsize=16,
#    ha='right',va='bottom',transform=ax.transData,rotation=0)
#ax.text(ax.get_xlim()[1],rapo,r'apo-center',color='k',fontsize=16,
#    ha='right',va='bottom',transform=ax.transData,rotation=0)
#ax.text(tfinal,ax.get_ylim()[1]*0.7,r'$t_\mathrm{final}$',color='k',
#    fontsize=16,ha='right',va='bottom',transform=ax.transData,
#    rotation=90)
ax.text(tdyn0,ax.get_ylim()[1]*0.7,r'$t_\mathrm{dyn,0}$',color='k',
    fontsize=16,ha='right',va='bottom',transform=ax.transData,
    rotation=90)
ax.text(2.*tdyn0,ax.get_ylim()[1]*0.7,r'$2t_\mathrm{dyn,0}$',color='k',
    fontsize=16,ha='right',va='bottom',transform=ax.transData,
    rotation=90)
ax.text(3.*tdyn0,ax.get_ylim()[1]*0.7,r'$3t_\mathrm{dyn,0}$',color='k',
    fontsize=16,ha='right',va='bottom',transform=ax.transData,
    rotation=90)
# annotation
ax.text(0.05,0.1,r'g)',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transAxes)
    
#---
ax = fig1.add_subplot(gs[1,3:])
ax.set_xlim(0.,3.)
ax.set_ylim(0.,40.)
ax.set_xlabel(r'$t$ [Gyr]')
ax.set_ylabel(r'$l$ [kpc]')
#ax.set_title(r'')
# tick and tick label positions
if tmax >=2.5: 
    start, end = 0., (tmax//2.5)*2.5
    major_ticks = np.arange(start, end, 2.)
    minor_ticks = np.arange(start, end, 0.5)
else:
    start, end = 0., (tmax//0.5)*0.5
    major_ticks = np.arange(start, end, 0.5)
    minor_ticks = np.arange(start, end, 0.1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
start, end = 0., (ax.get_ylim()[1]//10.+1)*10.
major_ticks = np.arange(start, end, 10.)
minor_ticks = np.arange(start, end, 2.)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks,minor=True)
# scale
#ax.set_xscale('log')
#ax.set_yscale('log')
# grid
#ax.grid(which='minor', alpha=0.2)                                                
#ax.grid(which='major', alpha=0.4)
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
#
# ax2 = ax.twinx()
# ax2.set_ylabel(r'$c_\mathrm{max}=l_\mathrm{v}/l_\mathrm{max}$',
#     color='dodgerblue')
# ax2.set_ylim(0.,100.)
# start, end = ax2.get_ylim()
# major_ticks = np.arange(start, end, 50.)
# minor_ticks = np.arange(start, end, 10.)
# ax2.set_yticks(major_ticks)
# ax2.set_yticks(minor_ticks,minor=True)
# #ax2.set_xscale('log')
# #ax2.set_yscale('log')
# ax2.tick_params('y',colors='dodgerblue',direction='in',
#     right='on',length=10,width=1,which='major')
# ax2.tick_params('y',colors='dodgerblue',direction='in',
#     right='on',length=5,width=1,which='minor')
#
ax2 = ax.twinx()
ax2.set_ylabel(r'$s_{0.01}=-\frac{\mathrm{d}\ln\rho}{\mathrm{d}\ln l}|_{l=0.01l_\mathrm{v}}$',
    color='dodgerblue')
ax2.set_ylim(0.,3.)
start, end = ax2.get_ylim()
major_ticks = np.arange(start, end, 0.5)
minor_ticks = np.arange(start, end, 0.1)
ax2.set_yticks(major_ticks)
ax2.set_yticks(minor_ticks,minor=True)
#ax2.set_xscale('log')
#ax2.set_yscale('log')
ax2.tick_params('y',colors='dodgerblue',direction='in',
    right='on',length=10,width=1,which='major')
ax2.tick_params('y',colors='dodgerblue',direction='in',
    right='on',length=5,width=1,which='minor')
#
# plot
ax.plot(timesteps,TidalRadius,lw=lw,color='k',label='tidal')
#ax.plot(timesteps,RamPressureRadius,lw=lw,ls='--',color='k',label='RP')
ax.plot(timesteps,10.*EffectiveRadius,lw=lw,ls='-.',color='k',
    label='$10\ l_\mathrm{eff}$')
#
#ax2.plot(timesteps,concentration,lw=lw,color='dodgerblue',)
ax2.plot(timesteps,slope,lw=lw,color='dodgerblue',)
#
# reference line
ax.plot(timesteps,np.repeat(0.1*lv0,Nstep),lw=lw,color='grey',ls=':')
ax.fill_between(timesteps, np.repeat(0.1*lv0,Nstep), 0.,
    facecolor='purple', alpha=0.3, lw=0,)  
#
#ax.plot(np.repeat(tfinal,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(2.*tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(3.*tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')
#
# legends
ax.legend(loc='best',numpoints=1,scatterpoints=1,fontsize=14,
    frameon=True) #bbox_to_anchor=(1.2, 0.9)
# annotation
ax.text(0.5*tmax,0.1*lv0,r'$0.1l_\mathrm{v}(0)$',color='k',
    fontsize=16,ha='right',va='bottom',transform=ax.transData,rotation=0)
# annotation
ax.text(0.05,0.1,r'h)',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transAxes)

#---
ax = fig1.add_subplot(gs[2,3:])
ax.set_xlim(0.,3.)
ax.set_ylim(1e-3*mv0,2.*mv0)
ax.set_xlabel(r'$t$ [Gyr]')
ax.set_ylabel(r'$m$ [$M_\odot$]')
#ax.set_title(r'')
# scale
#ax.set_xscale('log')
ax.set_yscale('log')
# tick and tick label positions
if tmax >=2.5: 
    start, end = 0., (tmax//2.5)*2.5
    major_ticks = np.arange(start, end, 2.)
    minor_ticks = np.arange(start, end, 0.5)
else:
    start, end = 0., (tmax//0.5)*0.5
    major_ticks = np.arange(start, end, 0.5)
    minor_ticks = np.arange(start, end, 0.1)
ax.set_xticks(major_ticks)
ax.set_xticks(minor_ticks,minor=True)
# for refined control of log-scale tick marks
locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax.yaxis.set_major_locator(locmaj)
ax.yaxis.set_minor_locator(locmin)
ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# grid
#ax.grid(which='minor', alpha=0.2)                                                
#ax.grid(which='major', alpha=0.4)
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
#
ax2 = ax.twinx()
ax2.set_ylabel(r'$\mathrm{d}m/\mathrm{d}t$ [$M_\odot/Gyr$]',color='green')
ax2.set_ylim(1e5,1e12)
#start, end = ax2.get_ylim()
#major_ticks = np.arange(start, end, 50.)
#minor_ticks = np.arange(start, end, 10.)
#ax2.set_yticks(major_ticks)
#ax2.set_yticks(minor_ticks,minor=True)
#ax2.set_xscale('log')
ax2.set_yscale('log')
# for refined control of log-scale tick marks
locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
locmin = mpl.ticker.LogLocator(base=10.0,subs=(0.2,0.4,0.6,0.8),numticks=12)
ax2.yaxis.set_major_locator(locmaj)
ax2.yaxis.set_minor_locator(locmin)
ax2.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
ax2.tick_params('y',colors='green',direction='in',
    right='on',length=10,width=1,which='major')
ax2.tick_params('y',colors='green',direction='in',
    right='on',length=5,width=1,which='minor')
#
# plot
ax.plot(timesteps,mass,lw=lw,color='k',label='total')
#ax.plot(timesteps,GasMass,lw=lw,ls='--',color='k',label='gas')
ax.plot(timesteps,StellarMass,lw=lw,ls='-.',color='k',label='star')
#
ax2.plot(timesteps,MassLossRate,lw=lw,color='green')
#ax2.plot(timesteps,GasLossRate,lw=lw,ls='--',color='green')
#
# reference line
#ax.plot(np.repeat(tfinal,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(2.*tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')
ax.plot(np.repeat(3.*tdyn0,2),ax.get_ylim(),lw=2,ls=":",color='k')
# legends
ax.legend(loc='best',numpoints=1,scatterpoints=1,fontsize=14,
    frameon=True) #bbox_to_anchor=(1.2, 0.9)
# annotation
ax.text(0.05,0.1,r'i)',color='k',fontsize=16,
    ha='left',va='bottom',transform=ax.transAxes)

#---save figure
plt.savefig(outfig1,dpi=300)
fig1.canvas.manager.window.raise_()
plt.get_current_fig_manager().window.setGeometry(50,50,1600,900)
fig1.show()

#sys.exit()

