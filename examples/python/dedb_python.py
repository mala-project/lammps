from __future__ import print_function
from lammps import lammps
import ctypes
import numpy as np
from numpy.random import RandomState
from ctypes import *

#NOTE you may import mala functions here
#import mala.  

#for now, a global grid dEdB is used to evaluate forces - this function gives the grid index per nx,ny,nz point
def get_grid(ngrid):
    igrid = 0
    for nx in range(ngrid):
        for ny in range(ngrid):
            for nz in range(ngrid):
                igrid += 1
    return igrid

def pre_force_callback(lmp):
    #NOTE inside this callback, you may access the lammps state:
    L = lammps(ptr=lmp)

    # flag to use contiguous dEdB (useful for casting numpy arrays in LAMMPS later)
    flat_beta = True
    #-------------------------------------------------------------
    # variables to access fix pointer in python if needed
    #-------------------------------------------------------------
    fid = '4'
    ftype = 2 # 0 for scalar 1 for vector 2 for array
    result_type = 2
    compute_style = 0
    fstyle = 0

    #-------------------------------------------------------------
    # variables to define grid 
    #-------------------------------------------------------------
    ncolbase = 0 # = 6 if accessing alocal from the fix
    nrow = (get_grid(ngrid=2)) # get the number of global grid points - dE/dB for each 
    ncoef = int(368/2) #number of ace descriptors per atom - may be obtained from ACE functions in mala
    ncol = ncoef + ncolbase
    base_array_rows = 0
    #-------------------------------------------------------------
    # dummy function to get dE_I/dB_{I,k} for a
    #-------------------------------------------------------------
    prng = RandomState(3481)
    betas_row = prng.uniform(-1,1,ncoef)*1.e-4 #np.arange(ncoef)*1.e-6
    betas = np.repeat(np.array([betas_row]),repeats = nrow+base_array_rows,axis=0)

    #If you need to access descriptors from lammps to evaluate betas,
    # a future update will allow you to extract them from the fix with:
    #this_fix = L.numpy.extract_fix(fid,fstyle,ftype,nrow,ncol)


    #NOTE that this accessing the descriptors from the fix so far seems to throw mem.
    #  errors - possibly due to python garbage collection
    # to get around this if this,we may access the descriptors through an extract_compute:
    #these_descriptors = L.numpy.extract_compute(cid,cstyle,ctype,crow,ccol)

    #-------------------------------------------------------------
    # this preforce callback should return dEdB in numpy array 
    #    format.
    #-------------------------------------------------------------
    if flat_beta:
        betas = betas.flatten()
        return np.ascontiguousarray(betas)
    else:
        return betas

