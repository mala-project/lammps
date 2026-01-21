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

#include "compute_pace_grid_local.h"

#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ace_evaluator.h"
#include "ace-evaluator/ace_types.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "memory.h"
#include "modify.h"
#include "update.h"

#include <math.h>
#include <cstring>

namespace LAMMPS_NS {
struct ACECLimpl {
  ACECLimpl() : basis_set(nullptr), ace(nullptr) {}
  ~ACECLimpl()
  {
    delete basis_set;
    delete ace;
  }
  ACECTildeBasisSet *basis_set;
  ACECTildeEvaluator *ace;
};
}    // namespace LAMMPS_NS

using namespace LAMMPS_NS;

ComputePACEGridLocal::ComputePACEGridLocal(LAMMPS *lmp, int narg, char **arg) :
    ComputeGridLocal(lmp, narg, arg), map(nullptr)
{
  // skip over arguments used by base class
  // so that argument positions are identical to
  // regular per-atom compute
  arg += nargbase;
  narg -= nargbase;


  int ntypes = atom->ntypes;
  int nargmin = 6 + 2 * ntypes;

  //if (narg < nargmin) error->all(FLERR, "Illegal compute {} command", style);


  aceclimpl = new ACECLimpl;

  bzeroflag = 1;
  nelements = 1;
  chunksize = 32768;
  parallel_thresh = 8192;


  //read in file with CG coefficients or c_tilde coefficients
  gridtypeflagl = 1;

  //read in file with CG coefficients or c_tilde coefficients
  char * potential_file_name = arg[3];
  int iarg = nargmin;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "ugridtype") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute {} command", style);
      gridtypeflagl = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    }
  }
  //auto potential_file_name = utils::get_potential_file_path(arg[3]);
  delete aceclimpl->basis_set;
  aceclimpl->basis_set = new ACECTildeBasisSet(potential_file_name);
  cutmax = aceclimpl->basis_set->cutoffmax;
  nelements = aceclimpl->basis_set->nelements;
  memory->create(map,nelements+1,"pace/grid/local:map");
  for (int ielemi = 0; ielemi <= nelements; ielemi++){
    map[ielemi]=0;
  }

  //# of rank 1, rank > 1 functions
  int ielem = 0;
  if (gridtypeflagl){
    ielem = nelements-1;
  }
  int n_r1, n_rp = 0;
  n_r1 = aceclimpl->basis_set->total_basis_size_rank1[ielem];
  n_rp = aceclimpl->basis_set->total_basis_size[ielem];

  int ncoeff = n_r1 + n_rp;

  nvalues = ncoeff;

  size_local_cols = size_local_cols_base + nvalues;
}


/* ---------------------------------------------------------------------- */

ComputePACEGridLocal::~ComputePACEGridLocal()
{
  if (copymode) return;

  delete aceclimpl;

  memory->destroy(map);
  if (grid_is_allocated) {
    memory->destroy(gridneigh);
    memory->destroy(gridinside);
    memory->destroy(gridtype);
  }
}

/* ---------------------------------------------------------------------- */

void ComputePACEGridLocal::init()
{
  if ((modify->get_compute_by_style("^pace/grid/local$").size() > 1) && (comm->me == 0))
    error->warning(FLERR, "More than one instance of compute pace/grid/local");

}

/* ---------------------------------------------------------------------- */

void ComputePACEGridLocal::grow_grid(int newnmax)
{
  //if (newnmax <= nmax) return;
  nmax=newnmax;
  if (grid_is_allocated) {
    memory->destroy(gridneigh);
    memory->destroy(gridinside);
    memory->destroy(gridtype);
  }
  memory->create(gridneigh, nmax, 3, "pace/grid/local:gridneigh");
  memory->create(gridinside, nmax, "pace/grid/local:gridinside");
  memory->create(gridtype, nmax, "pace/grid/local:gridtype");
  for (int jz = 0; jz < nmax; jz++){
    gridneigh[jz][0] = 0.00000000;
    gridneigh[jz][1] = 0.00000000;
    gridneigh[jz][2] = 0.00000000;
    gridinside[jz] = 0;
    gridtype[jz] = 0;
  }
  grid_is_allocated = 1;
}
/* ---------------------------------------------------------------------- */

void ComputePACEGridLocal::compute_local()
{

  invoked_local = update->ntimestep;

  // compute pace for each gridpoint

  double **const x = atom->x;
  const int *const mask = atom->mask;
  int *const type = atom->type;
  ntypes = atom->ntypes;
  const int ntotal = atom->nlocal + atom->nghost;

  // grow grid arrays to match possible # of grid neighbors (analogous to jnum)
  nmax = ntotal + 1;
  grow_grid(nmax);
  
  //for (int jzj = 0; jzj < nmax; jzj++){
  //  gridneigh[jzj][0] = 0.0;
  //  gridneigh[jzj][1] = 0.0;
  //  gridneigh[jzj][2] = 0.0;
  //}

  int igrid = 0;
  for (int iz = nzlo; iz <= nzhi; iz++)
    for (int iy = nylo; iy <= nyhi; iy++)
      for (int ix = nxlo; ix <= nxhi; ix++) {
        double xgrid[3];
        grid2x(ix, iy, iz, xgrid);
        const double xtmp = xgrid[0];
        const double ytmp = xgrid[1];
        const double ztmp = xgrid[2];

        // currently, all grid points are type 1
        // - later we may want to set up lammps_user_pace interface such that the
        // real atom types begin at 2 and reserve type 1 specifically for the
        // grid entries. This will allow us to only evaluate ACE for atoms around
        // a grid point, but with no contributions from other grid points
        int ielem = 0;
        if (gridtypeflagl){
          ielem = nelements-1;
        }
        const int itype = ielem+1;
        delete aceclimpl->ace;
        aceclimpl->ace = new ACECTildeEvaluator(*aceclimpl->basis_set);
        aceclimpl->ace->compute_projections = true;
        aceclimpl->ace->compute_b_grad = false;

        // leave these here b/c in the future, we may be able to allow
        // for different #s of descriptors per atom or grid type
        int n_r1, n_rp = 0;
        n_r1 = aceclimpl->basis_set->total_basis_size_rank1[ielem];
        n_rp = aceclimpl->basis_set->total_basis_size[ielem];
        int ncoeff = n_r1 + n_rp;
 
        //ACE element mapping       
        aceclimpl->ace->element_type_mapping.init(nelements+1);
        for (int imu=0; imu <= nelements; imu++){
          aceclimpl->ace->element_type_mapping(imu) = 0;
        }
        for (int ik = 1; ik <= nelements+1; ik++) {
          for(int mu = 0; mu < nelements; mu++){
            if (mu != -1) {
              if (mu == ik - 1) {
                map[ik] = mu;
                aceclimpl->ace->element_type_mapping(ik) = mu;
              }
            }
          }
        }

        // rij[][3] = displacements between atom I and those neighbors
        // inside = indices of neighbors of I within cutoff
        // typej = types of neighbors of I within cutoff

        // build short neighbor list
        // add in grid position and indices manually at end of short neighbor list
        
        grow_grid(nmax);
        ninside = 0;
        for (int j = 0; j < ntotal; j++) {
          if (!(mask[j] & groupbit)) continue;
          const double delx = xtmp - x[j][0];
          const double dely = ytmp - x[j][1];
          const double delz = ztmp - x[j][2];
          const double rsq = delx * delx + dely * dely + delz * delz;
          const double rxtmp = x[j][0];
          const double rytmp = x[j][1];
          const double rztmp = x[j][2];
          const double rsqx = pow(rxtmp,2) + pow(rytmp,2) + pow(rztmp,2);
          int jtype = type[j];
          int jelem = map[jtype];
          double thiscut = aceclimpl->basis_set->radial_functions->cut(ielem,jelem);
          const double rnorm = sqrt(rsq);
          const double rnormx = sqrt(rsqx);
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
        int ninsidegrid = ninside+1;
        int copy_igrid = igrid;
        const int baseind = 0;

        gridinside[ninside]=ninside;
        gridneigh[ninside][0] = xtmp;
        gridneigh[ninside][1] = ytmp;
        gridneigh[ninside][2] = ztmp;
        gridtype[ninside]=itype;

        // perform ACE evaluation with short neighbor list
        aceclimpl->ace->resize_neighbours_cache(ninside);
        aceclimpl->ace->compute_atom(ninside, gridneigh, gridtype, ninside, gridinside);
        Array1D<DOUBLE_TYPE> Bs = aceclimpl->ace->projections;
        for (int icoeff = 0; icoeff < ncoeff; icoeff++){
          alocal[igrid][size_local_cols_base + icoeff] = Bs(icoeff);
        }
        igrid++;

      }
}

/* ----------------------------------------------------------------------
   memory usage
------------------------------------------------------------------------- */

double ComputePACEGridLocal::memory_usage()
{
  //double nbytes = (double) size_array_rows * size_array_cols * sizeof(double);    // grid
  //nbytes += (double) size_array_rows * size_array_cols * sizeof(double);          // gridall
  //nbytes += (double) size_array_cols * ngridlocal * sizeof(double);               // gridlocal
  int n = atom->ntypes + 1;
  double nbytes = (double) n * sizeof(int);    // map
  nbytes += (double)nmax * 3 * sizeof(double);                    // gridneigh
  nbytes += (double)nmax * sizeof(int);                           // gridinside
  nbytes += (double)nmax * sizeof(int);                           // gridtype


  return nbytes;
}
