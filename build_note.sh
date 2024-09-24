#!/bin/bash

mkdir build && cd build
# Build and compile to use the fix python/gridforceace
#NOTE first add your numpy include dir to your cplus include path
export CPLUS_INCLUDE_PATH=$(python -c "import numpy; print(numpy.get_include())")

cmake ../cmake -DLAMMPS_EXCEPTIONS=yes \
               -DBUILD_SHARED_LIBS=yes \
               -DBUILD_MPI=no \
               -DBUILD_OMP=no \
               -DPKG_PYTHON=yes \
               -DPKG_ML-SNAP=yes \
               -DPKG_ML-PACE=yes \
               -DPKG_ML-IAP=yes \
               -DPKG_MLIAP_ENABLE_PYTHON=yes \
               -DPYTHON_EXECUTABLE:FILEPATH=`which python` \
               -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
               -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") 

make -j
make install-python

# To compile and build when only using the ace grid computes

cmake ../cmake -DLAMMPS_EXCEPTIONS=yes \
               -DBUILD_SHARED_LIBS=yes \
               -DBUILD_MPI=yes \
               -DBUILD_OMP=yes \
               -DPKG_PYTHON=yes \
               -DPKG_ML-SNAP=yes \
               -DPKG_ML-PACE=yes \
               -DPKG_ML-IAP=yes \
               -DPKG_MLIAP_ENABLE_PYTHON=yes \
               -DPYTHON_EXECUTABLE:FILEPATH=`which python` \
               -DPYTHON_INCLUDE_DIR=$(python -c "import sysconfig; print(sysconfig.get_path('include'))")  \
               -DPYTHON_LIBRARY=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))") 

make -j
make install-python

#NOTE that the following dflags were optional in both cases above:
#               -DPKG_ML-IAP=yes \
#               -DPKG_MLIAP_ENABLE_PYTHON=yes \

#tested with:
#cmake version 3.26.5
#Python 3.10.13
#Numpy 1.26.4
#gcc (GCC) 8.5.0
