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
from numpy.fft import fftn,ifftn
#from scipy.fftpack import fftn, ifftn
from scipy.integrate import simps

# define parameters.
# SCFT  parameters 
XN=20.0  # Flory-huggins parameters for AB diblock copolymer
fA=0.24   # Volume fraction of A block
# unit cell parameters
Nx,Ny,Nz=32,32,32 # grid number on each dimension.
Lx,Ly,Lz=4.63,4.63,4.63 # unit cell length on each dimension, scaled by Rg
dx=Lx/Nx
dy=Ly/Ny
dz=Lz/Nz
# chain discretization
Ns=100
ds=1.0/Ns
NA=int(Ns*fA)
NB=Ns-NA

Andsn_nim=5
wA_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
wB_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
dwA_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
dwB_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))

class Andsn_fields(object):
    ''' containing the preceding fields for anderson mixing"

    '''
    def __init__(self,Andsn_nim,Nx):
        self.wA_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
        self.wB_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
        self.dwA_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))
        self.dwB_save=np.zeros((Andsn_nim+1,Nx,Nx,Nx))


def simpson_int_pbc(f,Rx_grid):
    Nx=f.shape[0]
    f_x=np.zeros(Nx)
    f_xy=np.zeros((Nx,Nx))
    for i in np.arange(Nx):
        for j in np.arange(Nx):
            f_xy[i,j]=simps(np.append(f[i,j,:],f[i,j,0]),dx=Rx_grid[1]) 
    for i in np.arange(Nx):
        f_x[i]=simps(np.append(f_xy[i,:],f_xy[i,0]),dx=Rx_grid[1]) 
    integral=simps(np.append(f_x[:],f_x[0]),dx=Rx_grid[1]) 

    return integral


def AB_diblock_mde_solver(q_f,q_b,wA,wB,fA,Nx,Ns,NA,ds,Kx_grid):
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
    
    #local_data_r=np.zeros((Nx,Nx,Nx),dtype=float)
    local_data_c=np.zeros((Nx,Nx,Nx),dtype=complex)
    for s in np.arange(1,Ns): # solving forward propagator 
        if s<NA:
            local_data_c[:,:,:]=q_f[s-1,:,:,:]*exp_w[0,:,:,:]+0.0j # for A block
            local_data_c=fftn(local_data_c) 
            local_data_c[:,:,:]=local_data_c[:,:,:]*exp_k2[:,:,:] 
            local_data_c=ifftn(local_data_c)
            q_f[s,:]=local_data_c[:,:,:].real*exp_w[0,:,:,:]  # !!! Note that different from FFTW, in Scipy,a pair of fft/ifft transforms is already normalized 
        else:
            local_data_c[:,:,:]=q_f[s-1,:,:,:]*exp_w[1,:,:,:]+0.0j
            local_data_c=fftn(local_data_c) 
            local_data_c[:,:,:]=local_data_c[:,:,:]*exp_k2[:,:,:] 
            local_data_c=ifftn(local_data_c)
            q_f[s,:]=local_data_c[:,:,:].real*exp_w[1,:,:,:]
    for s in np.arange(1,Ns)[::-1]: # solving backward propagator 
        if s<NA:
            local_data_c[:,:,:]=q_b[s,:,:,:]*exp_w[0,:,:,:]+0.0j # for A block
            local_data_c=fftn(local_data_c) 
            local_data_c[:,:,:]=local_data_c[:,:,:]*exp_k2[:,:,:] 
            local_data_c=ifftn(local_data_c)
            q_b[s-1,:]=local_data_c[:,:,:].real*exp_w[0,:,:,:]  # !!! Note that different from FFTW, in Scipy,a pair of fft/ifft transforms is already normalized 
        else:
            local_data_c[:,:,:]=q_b[s,:,:,:]*exp_w[1,:,:,:]+0.0j
            local_data_c=fftn(local_data_c) 
            local_data_c[:,:,:]=local_data_c[:,:,:]*exp_k2[:,:,:] 
            local_data_c=ifftn(local_data_c)
            q_b[s-1,:]=local_data_c[:,:,:].real*exp_w[1,:,:,:]

    return

def propagator_to_density(q_f,q_b,wA,wB,Phi_A,Phi_B,bigQ,F_fh,F_tot,XN,fA,Ns,NA,Nx,Lx,Rx_grid): # compute the density field from propagators
    Mv=Lx**3 # Volume of unit cell
    ds=1.0/Ns
    ar=np.zeros((Nx,Nx,Nx)) # temporary 3d array 
    ar[:,:,:]=q_f[Ns-1,:,:,:]
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
   
    totden=simpson_int_pbc(ar,Rx_grid)/Mv
    Phi_A[:,:,:]=Phi_A[:,:,:]/totden  
    Phi_B[:,:,:]=Phi_B[:,:,:]/totden  
    ar[:,:,:]=XN*Phi_A[:,:,:]*Phi_B[:,:,:]
    ar[:,:,:]=ar[:,:,:]-wA[:,:,:]*Phi_A[:,:,:]-wB[:,:,:]*Phi_B[:,:,:]
    ar[:,:,:]=ar[:,:,:]+0.5*(wA[:,:,:]+wB[:,:,:])*(Phi_A[:,:,:]+Phi_B[:,:,:]-1.0)
    F_fh=simpson_int_pbc(ar,Rx_grid)/Mv
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
    lambda_t=0.1 # iteration step 
    if update_scheme==0 :
        SimpleMixing_AB(wA_tmp,wA,wB_tmp,wB,lambda_t,field_err)
        if ITR%10==0: print "field_err, iteration step",field_err,ITR 
    elif update_scheme==1 :
        AndersonMixing_AB(wA,wB,wA_tmp,wB_tmp,lambda_t,ITR,field_err)
        if ITR%10==0: print "field_err, iteration step",field_err,ITR 
    else :
        raise ValueError('Unkonwn update scheme for fields, only simple mxing (0) or Anderson mixing (1) supported now')

    return field_err

def SimpleMixing_AB(wA_tmp,wA,wB_tmp,wB,lambda_t,field_err):
    field_err[0]=np.max(np.abs(wA_tmp[:,:,:]-wA[:,:,:]))
    field_err[1]=np.max(np.abs(wB_tmp[:,:,:]-wB[:,:,:]))
    wA[:,:,:]=wA[:,:,:]+lambda_t*(wA_tmp[:,:,:]-wA[:,:,:])
    wB[:,:,:]=wB[:,:,:]+lambda_t*(wB_tmp[:,:,:]-wB[:,:,:])
    return 



def AndersonMixing_AB(wA,wB,wA_tmp,wB_tmp,lambda_t,ITR,field_err):
    Num_SimpleMixing=10 # the first 20 steps are simple mixing
    #Andsn_nim=5         # the number of reserved previous fields for anderson mixing
    global Andsn_nim
    print "Andsn_nim=",Andsn_nim
    global wA_save,wB_save
    global dwA_save,dwB_save
    Nx=wA.shape[0]
    w1A=np.zeros((Nx,Nx,Nx))
    w1B=np.zeros((Nx,Nx,Nx))
    w2A=np.zeros((Nx,Nx,Nx))
    w2B=np.zeros((Nx,Nx,Nx))
    wA_itr=np.zeros((Nx,Nx,Nx))
    wB_itr=np.zeros((Nx,Nx,Nx))
    dwA_itr=np.zeros((Nx,Nx,Nx))
    dwB_itr=np.zeros((Nx,Nx,Nx))
    if ITR > Num_SimpleMixing :
        Andsn_nim_tmp=np.min((ITR-Num_SimpleMixing,Andsn_nim))
        U_mn=np.zeros((Andsn_nim_tmp,Andsn_nim_tmp))
        V_m=np.zeros(Andsn_nim_tmp)
    else:
        Andsn_nim_tmp=0 

    if ITR <Num_SimpleMixing:
        SimpleMixing_AB(wA_tmp,wA,wB_tmp,wB,lambda_t,field_err)
    elif ITR==Num_SimpleMixing:
        wA_save[0,:,:,:]=wA_tmp[:,:,:]       
        wB_save[0,:,:,:]=wB_tmp[:,:,:]       
	dwA_save[0,:,:,:]=wA_save[0,:,:,:]-wA[:,:,:]       
        dwB_save[0,:,:,:]=wB_save[0,:,:,:]-wB[:,:,:]       
        SimpleMixing_AB(wA_tmp,wA,wB_tmp,wB,lambda_t,field_err)
    else :
        k_andsn=(ITR-Num_SimpleMixing)%(Andsn_nim+1)
        wA_save[k_andsn,:,:,:]=wA_tmp[:,:,:] 
	dwA_save[k_andsn,:,:,:]=wA_save[k_andsn,:,:,:]-wA[:,:,:]       
        wB_save[k_andsn,:,:,:]=wB_tmp[:,:,:] 
	dwB_save[k_andsn,:,:,:]=wB_save[k_andsn,:,:,:]-wB[:,:,:]       
        field_err[0]=np.vdot(dwA_save[k_andsn,:,:,:],dwA_save[k_andsn,:,:,:])  
        field_err[0]=field_err[0]/np.vdot(wA_save[k_andsn,:,:,:],wA_save[k_andsn,:,:,:])  
        field_err[1]=np.vdot(dwB_save[k_andsn,:,:,:],dwB_save[k_andsn,:,:,:])  
        field_err[1]=field_err[1]/np.vdot(wB_save[k_andsn,:,:,:],wB_save[k_andsn,:,:,:])  
        field_err[:]=np.sqrt(field_err)
        for m_andsn in np.arange(1,Andsn_nim_tmp+1): # 1~andsn_nim_tmp for m
            k_m=(k_andsn-m_andsn)%(Andsn_nim+1)
            w1A[:,:,:]=dwA_save[k_andsn,:,:,:]-dwA_save[k_m,:,:,:]
            w1B[:,:,:]=dwB_save[k_andsn,:,:,:]-dwB_save[k_m,:,:,:]
            for n_andsn in np.arange(m_andsn,Andsn_nim_tmp+1): # m_andsn~Andsn_nim_tmp for n
                k_n=(k_andsn-n_andsn)%(Andsn_nim+1)
                w2A[:,:,:]=dwA_save[k_andsn,:,:,:]-dwA_save[k_n,:,:,:]
                w2B[:,:,:]=dwB_save[k_andsn,:,:,:]-dwB_save[k_n,:,:,:]
# Old fortran convention, indx started from 1, here simply used the old code.
                U_mn[m_andsn-1,n_andsn-1]=np.vdot(w1A,w2A)+np.vdot(w1B,w2B)                      
         # symmetric matrix, adding the other half.
        for m_andsn in np.arange(1,Andsn_nim_tmp+1): # 1~andsn_nim_tmp for m
            for n_andsn in np.arange(1,m_andsn): # 1~m_andsn-1 for n
                U_mn[m_andsn-1,n_andsn-1]=U_mn[n_andsn-1,m_andsn-1]

        w1A[:,:,:]=dwA_save[k_andsn,:,:,:]
        w1B[:,:,:]=dwB_save[k_andsn,:,:,:]
        for m_andsn in np.arange(1,Andsn_nim_tmp+1): # 1~andsn_nim_tmp for m
            k_m=(k_andsn-m_andsn)%(Andsn_nim+1)
            w2A[:,:,:]=dwA_save[k_andsn,:,:,:]-dwA_save[k_m,:,:,:]
            w2B[:,:,:]=dwB_save[k_andsn,:,:,:]-dwB_save[k_m,:,:,:]
            V_m[m_andsn-1]=np.vdot(w2A,w1A)+np.vdot(w2B,w1B) 
         # solve the Ux=V linear equations
        V_m = np.linalg.solve(U_mn, V_m)
        for m_andsn in np.arange(1,Andsn_nim_tmp+1): # 1~andsn_nim_tmp for m
            k_m=(k_andsn-m_andsn)%(Andsn_nim+1)
            wA_itr[:,:,:]=wA_itr[:,:,:]+V_m[m_andsn-1]*(wA_save[k_m,:,:,:]-wA_save[k_andsn,:,:,:]) 
            dwA_itr[:,:,:]=dwA_itr[:,:,:]+V_m[m_andsn-1]*(dwA_save[k_m,:,:,:]- \
                           dwA_save[k_andsn,:,:,:]) 
            wB_itr[:,:,:]=wB_itr[:,:,:]+V_m[m_andsn-1]*(wB_save[k_m,:,:,:]-wB_save[k_andsn,:,:,:]) 
            dwB_itr[:,:,:]=dwB_itr[:,:,:]+V_m[m_andsn-1]*(dwB_save[k_m,:,:,:]- \
                           dwB_save[k_andsn,:,:,:]) 
        wA_itr[:,:,:]=wA_itr[:,:,:]+wA_save[k_andsn,:,:,:]
        dwA_itr[:,:,:]=dwA_itr[:,:,:]+dwA_save[k_andsn,:,:,:]
        wB_itr[:,:,:]=wB_itr[:,:,:]+wB_save[k_andsn,:,:,:]
        dwB_itr[:,:,:]=dwB_itr[:,:,:]+dwB_save[k_andsn,:,:,:]
# the final mixing of old and new field
        wA[:,:,:]=wA_itr[:,:,:]+lambda_t*dwA_itr[:,:,:]
        wB[:,:,:]=wB_itr[:,:,:]+lambda_t*dwB_itr[:,:,:]

    return


def scft_output(F_tot,F_fh,Phi_A,Phi_B):
    print "total free energy per unit volume is",F_tot
    print "Flory-huggins free energy per unit volume is",F_fh
    Nx=Phi_A.shape[0]
    slice_yz=np.zeros((Nx,Nx))
    slice_yz[:,:]=Phi_A[0,:,:]
    pl.imshow(slice_yz)
    pl.show()

    return
