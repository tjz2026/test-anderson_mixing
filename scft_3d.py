# -*- coding: utf-8 -*-
"""
This is a example code of 3d AB -diblock copolymer melts SCFT,  
for purpose of demenstrating the efficiency of Anderson mixing vs simple mixing scheme.


=========

Author:  Jiuzhou Tang 
3. Nov, 2016
"""

import sys
from timeit import default_timer as timer
import pylab as pl
import numpy as np
#from mayavi import mlab
import scipy.ndimage as ndimage
from scipy.fftpack import fft, ifft
from scft_utility import *

## define parameters.
## SCFT  parameters 
#XN=20.0  # Flory-huggins parameters for AB diblock copolymer
#fA=0.24   # Volume fraction of A block
## unit cell parameters
#Nx,Ny,Nz=32,32,32 # grid number on each dimension.
#Lx,Ly,Lz=4.63,4.63,4.63 # unit cell length on each dimension, scaled by Rg
#dx=Lx/Nx
#dy=Ly/Ny
#dz=Lz/Nz
## chain discretization
#Ns=100
#ds=1.0/Ns
#NA=int(Ns*fA)
#NB=Ns-NA

# init SCFT simulation
# create grid in both real space and K space,assuming Nx==Ny==Nz
Rx_grid=np.arange(Nx,dtype=float)*dx
d_kx=2*np.pi/Lx
Kx_grid=np.arange(Nx,dtype=float)
Kx_grid[:Nx/2+1]=Kx_grid[:Nx/2+1]*d_kx
Kx_grid[Nx/2+1:]=d_kx*(Nx-Kx_grid[Nx/2+1:])

# allocate the memeory for the fields and propagators
wA=np.zeros((Nx,Nx,Nx))  # chemical potential field of A
wB=np.zeros((Nx,Nx,Nx))
Phi_A=np.zeros((Nx,Nx,Nx)) # density of A
Phi_B=np.zeros((Nx,Nx,Nx))
q_f=np.zeros((Ns,Nx,Nx,Nx)) # forward propagator
q_b=np.zeros((Ns,Nx,Nx,Nx))

# output results 
F_tot=0.0 # total free energy per unit volume
F_fh=0.0  # Flory-huggins (1/V) \int dV XN*Phi_A*Phi_B
bigQ=0.0  # single chain partition function
 
# init the BCC initial fields
for x in np.arange(Nx):
    for y in np.arange(Nx):
        for z in np.arange(Nx):
            wA[x,y,z]=XN*(1.0-fA*(1+0.7*(np.cos(2.0*np.pi*x/Nx)*np.cos(2.0*np.pi*y/Ny) + \
            np.cos(2.0*np.pi*y/Ny)*np.cos(2.0*np.pi*z/Nz) + \
            np.cos(2.0*np.pi*x/Nx)*np.cos(2.0*np.pi*z/Nz))))
            wB[x,y,z]=XN-wA[x,y,z] 

# plot the initial chemical field (using mayavi)
#val=ndimage.zoom(val, 2)
#mlab.contour3d(wA)
#mlab.show()


# starting to run  SCFT loop.
Max_ITR=1000 # maximu SCFT iteration steps
# If using anderson mixing, we need to store the preceding Andsn_nim steps chemcial fields
Andsn_nim=5 # how many preceding steps of fields are used
update_scheme=1 # 0 for simple mixing, 1 for anderson mixing
#if update_scheme==1 : 
    #Andsn_fields Andsn_AB
    #wA_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
    #wB_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
    #dwA_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
    #dwB_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))

field_err=np.zeros(2)
error_tol=1.0e-4 # error tolerrence of fields
for ITR in np.arange(Max_ITR):
    AB_diblock_mde_solver(q_f,q_b,wA,wB,fA,Nx,Ns,NA,ds,Kx_grid) # modified diffusion equation solver for AB diblock copolymer (pesudo spetrum method)
    propagator_to_density(q_f,q_b,wA,wB,Phi_A,Phi_B,bigQ,F_fh,F_tot,XN,fA,Ns,NA,Nx,Lx,Rx_grid) # compute the density field from propagators
    field_err=field_update(update_scheme,XN,fA,wA,wB,Phi_A,Phi_B,Rx_grid,Nx,ITR)  # update field with simple mixing (0) or anderson mixing (1)
    if np.sum(np.abs(field_err))< error_tol:
        print "SCFT converged with iter_error=",np.sum(np.abs(field_err))
        break  
scft_output(F_tot,F_fh,Phi_A,Phi_B)    # output SCFT simulation info



