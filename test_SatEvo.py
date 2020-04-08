#########################################################################

# Test the satellites evolved from SatEvo.py

# Arthur Fangzhou Jiang 2020 Caltech

######################## set up the environment #########################

import config as cfg
import cosmo as co
import galhalo as gh
import init 
import aux

import numpy as np
import sys
import os
from multiprocessing import Pool, cpu_count

import matplotlib as mpl # must import before pyplot
#mpl.use('Qt5Agg')
mpl.use('TkAgg')
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['font.size'] = 16  
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

############################ user control ###############################

Nmax = 10000 # an estimate of the maximum number of branches per tree
phi_res = 1e-5 # m_res / M_0 for selecting subhalos

#---data
datadir = "./OUTPUT_SAT_20200304_StrippingEfficiency0.8/"

#---reference observation
M2L,L36,Re_SPARC = np.genfromtxt(
    '../UDG/SPARC_GS_MRT.mrt.txt',skip_header=50,
    usecols=(1,9,12),unpack=True)
M2L = 0.5 # <<< test
Ms_SPARC = M2L * L36 * 1e9 # [M_sun]

#---for plot
outfig1 = "./FIGURE/test_SatEvo.pdf"

alpha_symbol = 1 # transparency for data points
size = 10 # symbol size
edgewidth = 0.3 # symbool edge width
lw = 2.5 # line width
def fcolor(v,vmin=0.,vmax=5.,choice='z'):
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
    if choice == 'z': 
        cm = plt.cm.nipy_spectral
        norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)
        scalarMap = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
        return scalarMap.to_rgba(v),cm,norm

############################## compute ##################################

#---get the list of data files
files = []    
for filename in os.listdir(datadir):
    if filename.startswith('tree') and filename.endswith('.npz'): 
        files.append(os.path.join(datadir, filename))
files.sort()
Ntree = len(files)

#---

m0 = []
mpeak = []
ms0 = []
mspeak = []
le0 = []
NGTm0 = np.zeros((Ntree,Nmax),np.int) # cumulative subhalo mass functions
m0_sorted = np.zeros((Ntree,Nmax)) # sorted subhalo mass arrays
NGTms0 = np.zeros((Ntree,Nmax),np.int) # cumulative stellar mass functions
ms0_sorted = np.zeros((Ntree,Nmax)) # sorted stellar mass arrays
NLTr0 = np.zeros((Ntree,Nmax),np.int) # cumulative radial distribution 
r0_sorted = np.zeros((Ntree,Nmax))
for i,file in enumerate(files):
   
    #---load trees
    f = np.load(file)
    redshift = f['redshift']
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
    
    #---identify roots' redshift ids
    izroot = mass.argmax(axis=1)
    id = np.arange(mass.shape[0])
    
    #---
    M0 = mass[0,0]
    m = mass[:,0]
    ma = mass[id,izroot]
    ms = StellarMass[:,0]
    msa = StellarMass[id,izroot]
    le = StellarSize[:,0]
    ip = ParentID[:,0]
    xv = coordinates[:,0,:]
    r = np.zeros(m.size) - 99. # to store the host-centric distances
    
    isurv = np.where((m>phi_res*M0) & (m<M0))[0] # ids of surviving branches
    Nsurv = isurv.size
    
    # find the host-centric distances of the surviving branches -- for a
    # high-order branch, use that of its ultimate 1st-order host
    for isrv in isurv:
        
        j = isrv        
        while ip[j]>0: j = ip[j] 
        r[isrv] =  np.sqrt(xv[j,0]**2 + xv[j,2]**2)
    
    #---record
    m0.append(m[isurv])
    mpeak.append(ma[isurv])
    ms0.append(ms[isurv])
    mspeak.append(msa[isurv])
    le0.append(le[isurv])
    
    NGTm0[i,-Nsurv:] = (np.arange(Nsurv)+1)[::-1]
    m0_sorted[i,-Nsurv:] = np.sort(m[isurv])
    
    NGTms0[i,-Nsurv:] = (np.arange(Nsurv)+1)[::-1]
    ms0_sorted[i,-Nsurv:] = np.sort(ms[isurv])
    
    NLTr0[i,-Nsurv:] = (np.arange(Nsurv)+1)
    r0_sorted[i,-Nsurv:] = np.sort(r[isurv])
    
    print('    %s, %6i surviving branches'%(file,Nsurv))

m0 = np.concatenate(m0)
mpeak = np.concatenate(mpeak)
ms0 = np.concatenate(ms0)
mspeak = np.concatenate(mspeak)
le0 = np.concatenate(le0)

#---prepare for plots

# median cumulative subhalo mass function
m0_sorted_c50 = np.percentile(m0_sorted,50,axis=0)
msk = m0_sorted_c50 > 0.
m0_sorted_c50 = m0_sorted_c50[msk]
NGTm0_c50 = (np.arange(m0_sorted_c50.size)+1)[::-1]

# median cumulative stellar mass function
ms0_sorted_c50 = np.percentile(ms0_sorted,50,axis=0)
msk = ms0_sorted_c50 > 0.
ms0_sorted_c50 = ms0_sorted_c50[msk]
NGTms0_c50 = (np.arange(ms0_sorted_c50.size)+1)[::-1]

# median cumulative radial distribution 
r0_sorted_c50 = np.percentile(r0_sorted,50,axis=0,interpolation='higher')
msk = r0_sorted_c50 > 0.
r0_sorted_c50 = r0_sorted_c50[msk]
NLTr0_c50 = (np.arange(r0_sorted_c50.size)+1)

#---reference models
lgm_ref = np.linspace(4,15,101)
m_ref = 10**lgm_ref

ms_RP17_z0 = 10.**gh.lgMs_RP17(lgm_ref,z=0.)
ms_B13_z0 = 10.**gh.lgMs_B13(lgm_ref,z=0.)

################################ plots ##################################

print(">>> Plotting ...") 

# close all previous figure windows
plt.close('all')

#------------------------------------------------------------------------

fig1 = plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k') 
fig1.subplots_adjust(left=0.07, right=0.98,
    bottom=0.12, top=0.97,hspace=0.25, wspace=0.25)
gs = gridspec.GridSpec(2, 3) 
fig1.suptitle(r'')

#---

ax = fig1.add_subplot(gs[0,0])
ax.set_xlim(1e8,1e12)
ax.set_ylim(0.9,200.) 
ax.set_xlabel(r'$m$')
ax.set_ylabel(r'$N(>m)$')
#ax.set_title(r' ')
# tick and tick label positions
#start, end = ax.get_xlim()
#major_ticks = np.arange(start, end, 0.1)
#minor_ticks = np.arange(start, end, 0.05)
#ax.set_xticks(major_ticks)
#ax.set_xticks(minor_ticks,minor=True)
#start, end = ax.get_ylim()
# scale
ax.set_xscale('log')
ax.set_yscale('log')
# grid
#ax.grid(which='minor', alpha=0.2)                                                
#ax.grid(which='major', alpha=0.4)
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major')
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor')
# plot
for x,y in zip(m0_sorted,NGTm0):
    ax.plot(x,y,lw=1,color='grey')
ax.plot(m0_sorted_c50,NGTm0_c50,lw=lw,color='k')
# legend
#ax.legend(loc='best',fontsize='x-small',numpoints=1)

ax = fig1.add_subplot(gs[0,1])
ax.set_xlim(1e5,1e10)
ax.set_ylim(0.9,200.) 
ax.set_xlabel(r'$m_\star$')
ax.set_ylabel(r'$N(>m_\star)$')
#ax.set_title(r' ')
# tick and tick label positions
#start, end = ax.get_xlim()
#major_ticks = np.arange(start, end, 0.1)
#minor_ticks = np.arange(start, end, 0.05)
#ax.set_xticks(major_ticks)
#ax.set_xticks(minor_ticks,minor=True)
#start, end = ax.get_ylim()
# scale
ax.set_xscale('log')
ax.set_yscale('log')
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
for x,y in zip(ms0_sorted,NGTms0):
    ax.plot(x,y,lw=1,color='grey')
ax.plot(ms0_sorted_c50,NGTms0_c50,lw=lw,color='k')
# legend
#ax.legend(loc='best',fontsize='x-small',numpoints=1)

#---

ax = fig1.add_subplot(gs[1,0])
ax.set_xlim(1e8,1e12)
ax.set_ylim(1e5,1e11) 
ax.set_xlabel(r'$m$ [$M_\odot$]',fontsize=18)
ax.set_ylabel(r'$m_\star$ [$M_\odot$]',fontsize=18)
ax.set_title(r'')
# scale
ax.set_xscale('log')
ax.set_yscale('log')
# grid
ax.grid(which='minor', alpha=0.2)                                                
ax.grid(which='major', alpha=0.4)
# tick and tick label positions
# start, end = ax.get_xlim()
# major_ticks = np.arange(start, end, 0.5)
# minor_ticks = np.arange(start, end, 0.1)
# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks,minor=True)
# start, end = ax.get_ylim()
# major_ticks = np.arange(start, end, 0.5)
# minor_ticks = np.arange(start, end, 0.1)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks,minor=True)
# for refined control of log-scale tick marks
locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
locmin = mpl.ticker.LogLocator(base=10.0,
    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major',zorder=301)
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor',zorder=301)
# plot
ax.scatter(mpeak,mspeak,marker='o',s=size,
    facecolor='w',
    edgecolor='k',linewidth=edgewidth,alpha=alpha_symbol,rasterized=True,
    label=r'$m=m_\mathrm{peak}$')
ax.scatter(m0,ms0,marker='o',s=size,
    facecolor='k',
    edgecolor='k',linewidth=edgewidth,alpha=alpha_symbol,rasterized=True,
    label=r'm=m(z=0)')
# reference lines
ax.plot(m_ref,ms_RP17_z0,lw=2,color='k',label='RP+17')
ax.plot(m_ref,ms_B13_z0,lw=2,color='k',ls='--',label='B+13')
# annotations
#ax.text(0.1,0.9,r' ',
#    color='k',fontsize=16,ha='left',va='bottom',
#    transform=ax.transAxes,rotation=0)
# legend
ax.legend(loc='best',fontsize=16,frameon=True)

ax = fig1.add_subplot(gs[1,1])
ax.set_xlim(1e5,1e10)
ax.set_ylim(0.1,20.) 
ax.set_xlabel(r'$m_\mathrm{\star}$ [$M_\odot$]',fontsize=18)
ax.set_ylabel(r'$l_\mathrm{eff}$ [kpc]',fontsize=18)
ax.set_title(r'')
# scale
ax.set_xscale('log')
ax.set_yscale('log')
# grid
ax.grid(which='minor', alpha=0.2)                                                
ax.grid(which='major', alpha=0.4)
# tick and tick label positions
# start, end = ax.get_xlim()
# major_ticks = np.arange(start, end, 0.5)
# minor_ticks = np.arange(start, end, 0.1)
# ax.set_xticks(major_ticks)
# ax.set_xticks(minor_ticks,minor=True)
# start, end = ax.get_ylim()
# major_ticks = np.arange(start, end, 0.5)
# minor_ticks = np.arange(start, end, 0.1)
# ax.set_yticks(major_ticks)
# ax.set_yticks(minor_ticks,minor=True)
# for refined control of log-scale tick marks
locmaj = mpl.ticker.LogLocator(base=10,numticks=12) 
locmin = mpl.ticker.LogLocator(base=10.0,
    subs=(0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9),
    numticks=12)
ax.xaxis.set_major_locator(locmaj)
ax.xaxis.set_minor_locator(locmin)
ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
# tick length
ax.tick_params('both',direction='in',top='on',right='on',length=10,
    width=1,which='major',zorder=301)
ax.tick_params('both',direction='in',top='on',right='on',length=5,
    width=1,which='minor',zorder=301)
# plot
# observational relations for reference
x_edges = 10**np.linspace(np.log10(ax.get_xlim()[0]),np.log10(ax.get_xlim()[1]),18)
y_edges = 10**np.linspace(np.log10(ax.get_ylim()[0]),np.log10(ax.get_ylim()[1]),18)
counts = np.histogram2d(Ms_SPARC, Re_SPARC, bins=(x_edges, y_edges))[0]
ax.pcolormesh(x_edges, y_edges, counts.T,cmap=plt.cm.Blues)
#s5=ax.scatter(Ms_SPARC,Re_SPARC,marker='^',s=size*2,
#    facecolor='b',edgecolor='b',linewidth=edgewidth,
#    alpha=0.5)
#
ax.scatter(ms0,le0,marker='o',s=size,
    facecolor='k',
    edgecolor='k',linewidth=edgewidth,alpha=alpha_symbol,rasterized=True)
# reference lines
# ...
# annotations
ax.text(0.6,0.1,r'obs: SPARC',
    color='steelblue',
    fontsize=16,ha='left',va='bottom',
    transform=ax.transAxes,rotation=0)
# legend
#ax.legend(loc='best',fontsize=16,frameon=True)

ax = fig1.add_subplot(gs[1,2])
ax.set_xlim(5.,300.)
ax.set_ylim(0.9,500.) 
ax.set_xlabel(r'$r$ [kpc]')
ax.set_ylabel(r'$N(<r)$')
#ax.set_title(r' ')
# tick and tick label positions
#start, end = ax.get_xlim()
#major_ticks = np.arange(start, end, 0.1)
#minor_ticks = np.arange(start, end, 0.05)
#ax.set_xticks(major_ticks)
#ax.set_xticks(minor_ticks,minor=True)
#start, end = ax.get_ylim()
# scale
ax.set_xscale('log')
ax.set_yscale('log')
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
for x,y in zip(r0_sorted,NLTr0):
    ax.plot(x,y,lw=1,color='grey')
ax.plot(r0_sorted_c50,NLTr0_c50,lw=lw,color='k')
# legend
#ax.legend(loc='best',fontsize='x-small',numpoints=1)

plt.savefig(outfig1,dpi=300)
fig1.canvas.manager.window.attributes('-topmost', 1) 
plt.get_current_fig_manager().window.wm_geometry('+50+50')
fig1.show()