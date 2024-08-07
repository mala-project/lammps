from __future__ import print_function
from lammps import lammps
import ctypes
import numpy as np
from ctypes import *

#NOTE you may import mala functions here
#import mala.  

flat_beta = True
def get_grid(ngrid):
    igrid = 0
    for nx in range(ngrid):
        for ny in range(ngrid):
            for nz in range(ngrid):
                igrid += 1
    return igrid

from numpy.random import RandomState
def pre_force_callback(lmp):
    L = lammps(ptr=lmp)

    def _extract_compute_np(lmp, name, compute_style, result_type, array_shape=None):
        if array_shape is None:
            array_np = lmp.numpy.extract_compute(name,compute_style, result_type)
        else:
            ptr = lmp.extract_compute(name, compute_style, result_type)
            if result_type == 0:

                # no casting needed, lammps.py already works

                return ptr
            if result_type == 2:
                ptr = ptr.contents
            total_size = np.prod(array_shape)
            buffer_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_double * total_size))
            array_np = np.frombuffer(buffer_ptr.contents, dtype=float)
            array_np.shape = array_shape
        return array_np

    #-------------------------------------------------------------
    # variables to access fix pointer in python if needed
    #-------------------------------------------------------------
    #fid = 'python/gridforceace' # id for the fix
    fid = '4'
    ftype = 2 # 0 for scalar 1 for vector 2 for array
    result_type = 2
    compute_style = 0
    fstyle = 0
 
    ncolbase = 0
    nrow = (get_grid(ngrid=3)) #for now add a dE_I/dB_{I,K}row for ALL gridpoints (global)
    ncoef = int(368/2) #number of ace descriptors per atom - may be obtained from ACE functions in mala
    ncol = ncoef + ncolbase
    base_array_rows = 1
    #-------------------------------------------------------------
    # dummy function to get dE_I/dB_{I,k} for a
    #-------------------------------------------------------------
    prng = RandomState(3481)
    betas_row = prng.uniform(-1,1,ncoef)*1.e-2 #np.arange(ncoef)*1.e-6
    betas = np.repeat(np.array([betas_row]),repeats = nrow+base_array_rows,axis=0)

    #-------------------------------------------------------------
    if flat_beta:
        betas = betas.flatten()
        return np.ascontiguousarray(betas)
    else:
        return betas

