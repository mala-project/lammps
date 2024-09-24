/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/ Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "compute_pace_grid.h"

#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ace_evaluator.h"
#include "ace-evaluator/ace_types.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <cstring>

namespace LAMMPS_NS {
struct ACECGimpl {
  ACECGimpl() : basis_set(nullptr), ace(nullptr) {}
  ~ACECGimpl()
  {
    delete basis_set;
    delete ace;
  }
  ACECTildeBasisSet *basis_set;
  ACECTildeEvaluator *ace;
};
}    // namespace LAMMPS_NS

using namespace LAMMPS_NS;

ComputePACEGrid::ComputePACEGrid(LAMMPS *lmp, int narg, char **arg) :
    ComputeGrid(lmp, narg, arg), map(nullptr)
{
  // skip over arguments used by base class
  // so that argument positions are identical to
  // regular per-atom compute
  arg += nargbase;
  narg -= nargbase;

  // begin code common to all SNAP computes

  //double rfac0, rmin0;
  //int twojmax, switchflag, bzeroflag, bnormflag, wselfallflag;

  int ntypes = atom->ntypes;
  int nargmin = 4; //6 + 2 * ntypes;

  //if (narg < nargmin) error->all(FLERR, "Illegal compute {} command", style);

  // default values

  acecgimpl = new ACECGimpl;

  bzeroflag = 1;
  nelements = 1;
  chunksize = 32768;
  parallel_thresh = 8192;

  // process required arguments
  gridtypeflag = 1;

  //read in file with CG coefficients or c_tilde coefficients
  char * potential_file_name = arg[3];
  int iarg = nargmin;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "ugridtype") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute {} command", style);
      gridtypeflag = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    }
  }
  
  //auto potential_file_name = utils::get_potential_file_path(arg[3]);
  delete acecgimpl->basis_set;
  acecgimpl->basis_set = new ACECTildeBasisSet(potential_file_name);
  cutmax = acecgimpl->basis_set->cutoffmax;
  nelements = acecgimpl->basis_set->nelements;

  //# of rank 1, rank > 1 functions
  //assume ielem 0 (only mu0=0 basis functions for grid ACE)
  int ielem = 0;
  if (gridtypeflag){
    ielem = nelements-1;
  }
  int n_r1, n_rp = 0;
  n_r1 = acecgimpl->basis_set->total_basis_size_rank1[ielem];
  n_rp = acecgimpl->basis_set->total_basis_size[ielem];
  memory->create(map,nelements+1,"pace/grid:map");

  int ncoeff = n_r1 + n_rp;
  // set local input checks

  // process optional args


  nvalues = ncoeff;

  size_array_cols = size_array_cols_base + nvalues;
  array_flag = 1;
}

/* ---------------------------------------------------------------------- */

ComputePACEGrid::~ComputePACEGrid()
{
  if (copymode) return;

  delete acecgimpl;

  memory->destroy(map);
  if (ggrid_is_allocated) {
    memory->destroy(gridneigh);
    memory->destroy(gridinside);
    memory->destroy(gridtype);
  }
}

/* ---------------------------------------------------------------------- */

void ComputePACEGrid::init()
{
  if ((modify->get_compute_by_style("^pace/grid$").size() > 1) && (comm->me == 0))
    error->warning(FLERR, "More than one instance of compute pace/grid");

}

/* ---------------------------------------------------------------------- */
void ComputePACEGrid::grow_ggrid(int newnmax)
{
  //if (newnmax <= nmax) return;
  nmax=newnmax;
  if (ggrid_is_allocated) {
    memory->destroy(gridneigh);
    memory->destroy(gridinside);
    memory->destroy(gridtype);
  }
  memory->create(gridneigh, nmax, 3, "pace/grid:gridneigh");
  memory->create(gridinside, nmax, "pace/grid:gridinside");
  memory->create(gridtype, nmax, "pace/grid:gridtype");
  for (int jz = 0; jz < nmax; jz++){
    gridneigh[jz][0] = 0.00000000;
    gridneigh[jz][1] = 0.00000000;
    gridneigh[jz][2] = 0.00000000;
    gridinside[jz] = 0;
    gridtype[jz] = 0;
  }
  ggrid_is_allocated = 1;
}
/* ---------------------------------------------------------------------- */

void ComputePACEGrid::compute_array()
{

  invoked_array = update->ntimestep;

  // compute pace for each gridpoint

  double **const x = atom->x;
  const int *const mask = atom->mask;
  int *const type = atom->type;
  int ntypes = atom->ntypes;
  const int ntotal = atom->nlocal + atom->nghost;
  nmax = ntotal + 1;

  grow_ggrid(nmax);

  // loop over grid
  for (int iz = nzlo; iz <= nzhi; iz++)
    for (int iy = nylo; iy <= nyhi; iy++)
      for (int ix = nxlo; ix <= nxhi; ix++) {
        double xgrid[3];
        const int igrid = iz * (nx * ny) + iy * nx + ix;
        grid2x(igrid, xgrid);
        const double xtmp = xgrid[0];
        const double ytmp = xgrid[1];
        const double ztmp = xgrid[2];

        // currently, all grid points are type 1
        // - later we may want to set up lammps_user_pace interface such that the
        // real atom types begin at 2 and reserve type 1 specifically for the
        // grid entries. This will allow us to only evaluate ACE for atoms around
        // a grid point, but with no contributions from other grid points

        int ielem = 0;
        if (gridtypeflag){
          ielem = nelements-1;
        }
        const int itype = ielem+1;
        delete acecgimpl->ace;
        acecgimpl->ace = new ACECTildeEvaluator(*acecgimpl->basis_set);
        acecgimpl->ace->compute_projections = true;
        acecgimpl->ace->compute_b_grad = false;

        // leave these here b/c in the future, we may be able to allow
        // for different #s of descriptors per atom or grid type
        int n_r1, n_rp = 0;
        n_r1 = acecgimpl->basis_set->total_basis_size_rank1[ielem];
        n_rp = acecgimpl->basis_set->total_basis_size[ielem];
        int ncoeff = n_r1 + n_rp;

        //ACE element mapping
        acecgimpl->ace->element_type_mapping.init(nelements+1);
        for (int imu=0; imu <= nelements; imu++){
          acecgimpl->ace->element_type_mapping(imu) = 0;
        }
        for (int ik = 1; ik <= nelements+1; ik++) {
          for(int mu = 0; mu < nelements; mu++){
            if (mu != -1) {
              if (mu == ik - 1) {
                map[ik] = mu;
                acecgimpl->ace->element_type_mapping(ik) = mu;
              }
            }
          }
        }

        // rij[][3] = displacements between atom I and those neighbors
        // inside = indices of neighbors of I within cutoff
        // typej = types of neighbors of I within cutoff

        // build short neighbor list
        // add in grid position and indices manually at end of short neighbor list

        grow_ggrid(nmax);

        int ninside = 0;
        for (int j = 0; j < ntotal; j++) {
          if (!(mask[j] & groupbit)) continue;
          const double delx = xtmp - x[j][0];
          const double dely = ytmp - x[j][1];
          const double delz = ztmp - x[j][2];
          const double rsq = delx * delx + dely * dely + delz * delz;
          const double rnorm = sqrt(rsq);
          int jtype = type[j];
          int jelem = map[jtype];
          double thiscut = acecgimpl->basis_set->radial_functions->cut(ielem,jelem);
          if (rnorm < (thiscut) && rnorm > 1.e-20) {
            gridneigh[ninside][0] = x[j][0];
            gridneigh[ninside][1] = x[j][1];
            gridneigh[ninside][2] = x[j][2];
            gridinside[ninside] = ninside;
            gridtype[ninside] = jtype;
            ninside++;
          }
        }
        //add in grid site at final index
        gridinside[ninside]=ninside;
        gridneigh[ninside][0] = xtmp;
        gridneigh[ninside][1] = ytmp;
        gridneigh[ninside][2] = ztmp;
        gridtype[ninside]=itype;

        // perform ACE evaluation with short neighbor list
        acecgimpl->ace->resize_neighbours_cache(ninside);
        acecgimpl->ace->compute_atom(ninside, gridneigh, gridtype, ninside, gridinside);
        Array1D<DOUBLE_TYPE> Bs = acecgimpl->ace->projections;
        for (int icoeff = 0; icoeff < ncoeff; icoeff++){
          gridlocal[size_array_cols_base + icoeff][iz][iy][ix] = Bs(icoeff);
        }

      }

  memset(&grid[0][0], 0, sizeof(double) * size_array_rows * size_array_cols);

  for (int iz = nzlo; iz <= nzhi; iz++)
    for (int iy = nylo; iy <= nyhi; iy++)
      for (int ix = nxlo; ix <= nxhi; ix++) {
        const int igrid = iz * (nx * ny) + iy * nx + ix;
        for (int j = 0; j < nvalues; j++)
          grid[igrid][size_array_cols_base + j] = gridlocal[size_array_cols_base + j][iz][iy][ix];
      }
  MPI_Allreduce(&grid[0][0], &gridall[0][0], size_array_rows * size_array_cols, MPI_DOUBLE, MPI_SUM,
                world);
  assign_coords_all();
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputePACEGrid::memory_usage()
{
  double nbytes = (double) size_array_rows * size_array_cols * sizeof(double);    // grid
  nbytes += (double) size_array_rows * size_array_cols * sizeof(double);          // gridall
  nbytes += (double) size_array_cols * ngridlocal * sizeof(double);               // gridlocal
  nbytes += (double)nmax * 3 * sizeof(double);                    // gridneigh
  nbytes += (double)nmax * sizeof(int);                           // gridinside
  nbytes += (double)nmax * sizeof(int);                           // gridtype
  int n = atom->ntypes + 1;
  nbytes += (double) n * sizeof(int);    // map

  return nbytes;
}
