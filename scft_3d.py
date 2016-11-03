# -*- coding: utf-8 -*-
"""
This is a example code of 3d AB -diblock copolymer melts SCFT,  for purpose of demenstrating the efficiency of Anderson mixing vs simple mixing scheme.


=========

Author:  Jiuzhou Tang 
3. Nov, 2016
"""

import sys
from timeit import default_timer as timer
import pylab as pl
import numpy as np
from scipy.fftpack import fft, ifft


# define parameters.
# SCFT  parameters 
XN=20.0  # Flory-huggins parameters for AB diblock copolymer
fA=0.25   # Volume fraction of A block
# unit cell parameters
Nx,Ny,Nz=32,32,32 # grid number on each dimension.
Lx,Ly,Lz=4.63,4.63,4.63 # unit cell length on each dimension, scaled by Rg
dx=Lx/Nx
dy=Ly/Ny
dz=Lz/Nz
# chain discretization
Ns=200
ds=1.0/Ns
NA=int(Ns*fA)
NB=Ns-NA
# create grid in both real space and K space,assuming Nx==Ny==Nz
Rx_grid=np.arange(Nx,dtype=float)*dx
d_kx=2*np.pi/Lx
Kx_grid=np.arange(Nx,dtype=float)
Kx_grid[:Nx/2+1]=kgrid[:Nx/2+1]*d_kx
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

# starting to run  SCFT loop.
Max_ITR=1000 # maximu SCFT iteration steps
for ITR in np.arange(Max_ITR):
    AB_diblock_mde_solver(q_f,q_b,wA,wB,fA,Nx,NA,ds) # modified diffusion equation solver for AB diblock copolymer (pesudo spetrum method)
    propagator_to_density(q_f,q_b,fA,NA,NB) # compute the density field from propagators
    field_update(0)  # update field with simple mixing (0) or anderson mixing (1)
    if iter_error()< error_tol:
        print "SCFT converged with iter_error=",iter_error
        break  
scft_output()    # output SCFT simulation info



def AB_diblock_mde_solver(q_f,q_b,wA,wB,fA,Nx,NA,ds):
    q_f[0,:,:,:]=1.0 # initial condition of forward propagator
    q_b[Ns-1,:,:,:]=1.0 # initial condition of forward propagator
    exp_w=np.zeros((2,Nx,Nx,Nx))
    exp_k2=np.zeros((Nx,Nx,Nx))
    exp_w[0,:,:,:]=np.exp(wA[:,:,:]*(-0.5*ds))
    exp_w[1,:,:,:]=np.exp(wB[:,:,:]*(-0.5*ds))
    for x in np.arange(Nx):
        for y in np.arange(Nx):
            for z in np.arange(Nx):
                k2=Kx_grid[x]**2+Kx_grid[y]**2+Kx_grid[z]**2
                exp_k2[x,y,z]=np.exp(-1.0*ds*k2)
    
    local_data_r=np.zeros((Nx,Nx,Nx),dtype=float)
    local_data_c=np.zeros((Nx,Nx,Nx),dtype=complex)
    for s in np.arange(1,Ns): # solving forward propagator 
        if s<NA:
            local_data_r[:,:,:]=q_f[s-1,:,:,:]*exp_w[0,:,:,:] # for A block
            local_data_c=fftn(local_data_r) 
            local_data_c[:,:,:]=local_data_c[:,:,:]*exp_k2[:,:,:] 
            local_data_r=ifftn(local_data_c)
            q_f[s,:]=local_data_r[:,:,:]*exp_w[0,:,:,:]  # !!! Note that different from FFTW, in Scipy,a pair of fft/ifft transforms is already normalized 
        else:
            local_data_r[:,:,:]=q_f[s-1,:,:,:]*exp_w[1,:,:,:]
            local_data_c=fftn(local_data_r) 
            local_data_c[:,:,:]=local_data_c[:,:,:]*exp_k2[:,:,:] 
            local_data_r=ifftn(local_data_c)
            q_f[s,:]=local_data_r[:,:,:]*exp_w[1,:,:,:]
    for s in np.arange(1,Ns)[::-1]: # solving backward propagator 
        if s<NA:
            local_data_r[:,:,:]=q_b[s,:,:,:]*exp_w[0,:,:,:] # for A block
            local_data_c=fftn(local_data_r) 
            local_data_c[:,:,:]=local_data_c[:,:,:]*exp_k2[:,:,:] 
            local_data_r=ifftn(local_data_c)
            q_b[s-1,:]=local_data_r[:,:,:]*exp_w[0,:,:,:]  # !!! Note that different from FFTW, in Scipy,a pair of fft/ifft transforms is already normalized 
        else:
            local_data_r[:,:,:]=q_b[s,:,:,:]*exp_w[1,:,:,:]
            local_data_c=fftn(local_data_r) 
            local_data_c[:,:,:]=local_data_c[:,:,:]*exp_k2[:,:,:] 
            local_data_r=ifftn(local_data_c)
            q_b[s-1,:]=local_data_r[:,:,:]*exp_w[1,:,:,:]

    return

def propagator_to_density(q_f,q_b,fA,NA,NB): # compute the density field from propagators








  



