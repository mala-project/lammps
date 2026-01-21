/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef COMPUTE_CLASS
// clang-format off
ComputeStyle(pace/grid/local,ComputePACEGridLocal);
// clang-format on
#else

#ifndef LMP_COMPUTE_PACE_GRID_LOCAL_H
#define LMP_COMPUTE_PACE_GRID_LOCAL_H

#include "compute_grid_local.h"

namespace LAMMPS_NS {

class ComputePACEGridLocal : public ComputeGridLocal {
 public:
  ComputePACEGridLocal(class LAMMPS *, int, char **);
  ~ComputePACEGridLocal() override;
  void init() override;
  void compute_local() override;
  double memory_usage() override;
  void grow_grid(int);
  int ntypes;
  int nelements;
  int nmax=0;
  int ninside=0;
  int grid_is_allocated = 0;
  int gridtypeflagl;
  int ncoeff; // public for kokkos, but could go in the protected block now

 protected:
  //int ncoeff;
  double ** gridneigh;
  int *gridinside;
  int *gridtype;
  int *map;    // map types to [0,nelements)
  int parallel_thresh;
  double cutmax;
  int bzeroflag;
  int chunksize;

  struct ACECLimpl *aceclimpl;
};

}    // namespace LAMMPS_NS

#endif
#endif
