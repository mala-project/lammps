/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(python/acegridforce,FixPythonAceGridForce);
FixStyle(python,FixPythonAceGridForce);
// clang-format on
#else

#ifndef LMP_FIX_PYTHON_ACEGRIDFORCE_H
#define LMP_FIX_PYTHON_ACEGRIDFORCE_H

#include "fix.h"
#include <Python.h>   // IWYU pragma: export

namespace LAMMPS_NS {

class FixPythonAceGridForce : public Fix {
 public:
  FixPythonAceGridForce(class LAMMPS *, int, char **);
  ~FixPythonAceGridForce() override;
  int setmask() override;
  void end_of_step() override;
  void post_force(int) override;
  void pre_force(int) override;
  double compute_scalar() override;                // evaluate energy

 protected:
  void base_allocate();                // allocate pairstyle arrays
  void allocate_grid();                // create grid arrays
  void allocate_py_beta();             // create beta arrays
  void allocate_global();              //create global grid arrays
  void grow_grid(int);                 // grow pace grid arrays
  void deallocate_grid();              // free grid arrays
  void deallocate_py_beta();           // free global python beta arrays
  void deallocate_short();             // free short neighborlist arrays
  void deallocate_global();            // free global grid arrays
  void grid2x(int, int, int, double*); // convert local indices to coordinates
  void grid2xglobal(int,double*);      // convert global indices to coords
  void set_grid_global();              // set global grid
  void set_grid_local();               // set bounds for local grid
  void assign_coords();                // assign coords for grid
  void copy_gridlocal_to_local_array();// copy 4d gridlocal array to 2d local array
  void compute_beta();                 // get betas from someplace
  void process_pyarr(PyObject*);
  void compute(int, int);              // get descriptors for grid & evaluate E, F

  int nx, ny, nz;                      // global grid dimensions
  int nxlo, nxhi, nylo, nyhi, nzlo, nzhi; // local grid bounds, inclusive
  int ngridlocal;                      // number of local grid points
  int ngridglobal;                     // number of global grid points
  int nvalues;                         // number of values per grid point
  double ****gridlocal;                // local grid, redundant w.r.t. alocal
  double **alocal;                     // pointer to local array
  int triclinic;                       // triclinic flag
  double *boxlo, *prd;                 // box info (units real/ortho or reduced/tri)
  double *sublo, *subhi;               // subdomain info (units real/ortho or reduced/tri)
  double delxinv,delyinv,delzinv;      // inverse grid spacing
  double delx,dely,delz;               // grid spacing
  double **beta;                       // lammps-generated betas for all local grid points in list
  double **py_beta;                    // betas for all local grid points in list
  double *py_beta_contig;              // contiguous beta array from python
  double *e_grid;                      // energy per grid site (global)
  double *e_grid_all;                  // energy for all grid sites (global) 
  double cutmax;                       // max cutoff for radial functions
  int ndesc;                           // # of descriptors
  int ndesc_base;                      // base # of descriptors
  struct ACEFimpl *acefimpl;           // ace grid implementation from external c++ code
  double ** gridneigh;
  int *gridinside;
  int *gridtype;

 private:
  void *lmpPtr;
  void *pFunc;
  int selected_callback;
  int base_allocated;
  int short_allocated; //allocated short neighbor lists?
  int allocated_global;
  int allocated_py_beta;
  int gridlocal_allocated;
  int *map;                            // map types to [0,nelements)
  int gridtypeflagl;                   // flag to use unique type for ACE grid points
  int base_array_rows;
  bigint lasttime;
  int nelements;
  int nmax;
  int size_global_array_rows;               // rows for global grid
  int size_global_array_cols;               // columns for global grid

};

}    // namespace LAMMPS_NS

#endif
#endif
