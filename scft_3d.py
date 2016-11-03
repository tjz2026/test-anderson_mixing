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
update_scheme=0 # 0 for simple mixing, 1 for anderson mixing
field_err=np.zeros(2)
error_tol=1.0e-4 # error tolerrence of fields
for ITR in np.arange(Max_ITR):
    AB_diblock_mde_solver(q_f,q_b,wA,wB,fA,Nx,NA,ds) # modified diffusion equation solver for AB diblock copolymer (pesudo spetrum method)
    propagator_to_density(q_f,q_b,fA,NA,NB) # compute the density field from propagators
    field_err=field_update(update_scheme,XN,fA,wA,wB,Phi_A,Phi_B,Rx_grid,Nx,ITR)  # update field with simple mixing (0) or anderson mixing (1)
    if np.sum(np.abs(field_err))< error_tol:
        print "SCFT converged with iter_error=",np.sum(np.abs(field_err))
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

def propagator_to_density(q_f,q_b,Phi_A,Phi_B,bigQ,F_fh,F_tot,fA,Ns,NA,Nx,Lx): # compute the density field from propagators
    Mv=Lx**3 # Volume of unit cell
    ar=np.zeros(Nx,Nx,Nx)) # temporary 3d array 
    ar[:,:,:]=qf[Ns-1,:,:,:]
    bigQ=simpson_int_pbc(ar,Rx_grid)/Mv # simpson integral of a periodic unit cell
    Rs=np.zeros(Ns)
    sum_blk=0.0
    for x in np.arange(Nx) :
        for y in np.arange(Nx) :  
            for z in np.arange(Nx) :
                ar[x,y,z]=0.0
                Rs[0:NA]=q_f[0:NA,x,y,z]*q_b[0:NA,x,y,z] # summation of 0~NA-1 segments
                Phi_A[x,y,z]=simps(Rs[0:NA],dx=ds)
                Rs[NA:Ns]=q_f[NA:Ns,x,y,z]*q_b[NA:Ns,x,y,z] # summation of NA~Ns-1 segments
                Phi_B[x,y,z]=simps(Rs[NA:Ns-1],dx=ds)
                ar[x,y,z]=ar[x,y,z]+Phi_A[x,y,z]+Phi_B[x,y,z]
   
    totden=simposon_int_pbc(ar,Rx_grid)/Mv
    Phi_A[:,:,:]=Phi_A[:,:,:]/totden  
    Phi_B[:,:,:]=Phi_B[:,:,:]/totden  
    ar[:,:,:]=XN*Phi_A[:,:,:]*Phi_B[:,:,:]
    ar[:,:,:]=ar[:,:,:]-wA[:,:,:]*Phi_A[:,:,:]-wB[:,:,:]*Phi_B[:,:,:]
    ar[:,:,:]=ar[:,:,:]+0.5*(wA[:,:,:]+wB[:,:,:])*(Phi_A[:,:,:]+Phi_B[:,:,:]-1.0)
    F_fh=simposon_int_pbc(ar,Rx_grid)/Mv
    F_tot=F_fh-np.log(bigQ)
    return

    
def field_update(update_scheme,XN,fA,wA,wB,Phi_A,Phi_B,Rx_grid,Nx,ITR): # update the fields with the calculated density field, 0 for simple mixing, 1 for anderson mixing.
    wA_tmp=np.zeros((Nx,Nx,Nx))
    wB_tmp=np.zeros((Nx,Nx,Nx))
    yita=np.zeros((Nx,Nx,Nx))
    yita[:,:,:]=0.5*(wA[:,:,:]+wB[:,:,:])
    field_err=np.zeros(2)
    wA_tmp[:,:,:]=XN*(Phi_B[:,:,:]-(1.0-fA))+yita[:,:,:]
    wB_tmp[:,:,:]=XN*(Phi_A[:,:,:]-fA)+yita[:,:,:]
    if update_scheme==0 :
        SimpleMixing_AB(wA_temp,wA,wB_temp,wB,lambda_t,field_err)
        if ITR%10==0: print "field_err, iteration step",field_err,ITR 
    elif update_scheme==1 :
        AndersonMixing_AB()
    else :
        raise ValueError('Unkonwn update scheme for fields, only simple mxing (0) or Anderson mixing (1) supported now')

    return field_err

def SimpleMixing_AB(wA_temp,wA,wB_temp,wB,lambda_t,field_err):
    field_err[0]=np.max(np.abs(wA_tmp[:,:,:]-wA[:,:,:]))
    field_err[1]=np.max(np.abs(wB_tmp[:,:,:]-wB[:,:,:]))
    wA[:,:,:]=wA[:,:,:]+lambda_t*(wA_tmp[:,:,:]-wA[:,:,:])
    wB[:,:,:]=wB[:,:,:]+lambda_t*(wB_tmp[:,:,:]-wB[:,:,:])
    return 

















                        

