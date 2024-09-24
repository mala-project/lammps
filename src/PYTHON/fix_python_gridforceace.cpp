// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: James Goff (Sandia National Laboratories)
------------------------------------------------------------------------- */

#include "fix_python_gridforceace.h"

#include "ace-evaluator/ace_c_basis.h"
#include "ace-evaluator/ace_evaluator.h"
#include "ace-evaluator/ace_types.h"

// force/pair includes
#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "memory.h"

// python fix includes
#include "error.h"
#include "lmppython.h"
#include "python_compat.h"
#include "python_utils.h"
#include "update.h"
#include <numpy/arrayobject.h>

#include <cstring>
//#include <Python.h>   // IWYU pragma: export


// defines for random c++ betas
#define BETA_CONST 1.0e-5
#define BETA_IGRID 1.0e-6
#define BETA_ICOL 1.0e-6
#define INTERFACE_NUMPY 1
#define MPI_NUMPY 0

// define for linearized energy calculation
#define ENERGY_ROW 1

namespace LAMMPS_NS {
struct ACEFimpl {
  ACEFimpl() : basis_set(nullptr), ace(nullptr) {}
  ~ACEFimpl()
  {
    delete basis_set;
    delete ace;
  }
  ACECTildeBasisSet *basis_set;
  ACECTildeEvaluator *ace;
};
}    // namespace LAMMPS_NS

using namespace LAMMPS_NS;
using namespace FixConst;

/* ---------------------------------------------------------------------- */

FixPythonAceGridForce::FixPythonAceGridForce(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg), gridlocal(nullptr), alocal(nullptr), acefimpl(nullptr), py_beta(nullptr), py_beta_contig(nullptr),
  e_grid(nullptr), e_grid_all(nullptr)
{

  energy_global_flag = 1;

  if (narg < 6) error->all(FLERR,"Illegal fix python/acegridforce command");

  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Illegal fix python/acegridforce command");

  // ensure Python interpreter is initialized
  python->init();

  if (strcmp(arg[4],"post_force") == 0) {
    selected_callback = POST_FORCE;
  } else if (strcmp(arg[4],"end_of_step") == 0) {
    selected_callback = END_OF_STEP;
  } else if (strcmp(arg[4],"pre_force") == 0) {
    selected_callback = PRE_FORCE;
  } else {
    error->all(FLERR,"Unsupported callback name for fix python/acegridforce");
  }

  base_allocated = 0;
  size_global_array_rows = 0;
  gridlocal_allocated = 0;
  allocated_py_beta = 0;
  allocated_global = 0;
  short_allocated=0;
  gridtypeflagl = 1;
  nelements = 1;

  int nargmin = 11;
  acefimpl = new ACEFimpl;

  int iarg0 = 6;
  int iarg = iarg0;
  if (strcmp(arg[iarg], "grid") == 0) {
    if (iarg + 4 > narg) error->all(FLERR, "Illegal fix python/gridforceace grid command");
    nx = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
    ny = utils::inumeric(FLERR, arg[iarg + 2], false, lmp);
    nz = utils::inumeric(FLERR, arg[iarg + 3], false, lmp);
    if (nx <= 0 || ny <= 0 || nz <= 0) error->all(FLERR, "All grid dimensions must be positive");
    iarg += 4;
  } else
    error->all(FLERR, "Illegal fix python/gridforceace grid command");

  ngridglobal = nx * ny * nz;
  base_array_rows = 0;
#if ENERGY_ROW 
  base_array_rows = 1;
#endif
  size_global_array_rows = ngridglobal + base_array_rows;
  char * potential_file_name = arg[10];

  iarg = nargmin;

  while (iarg < narg) {
    if (strcmp(arg[iarg], "ugridtype") == 0) {
      if (iarg + 2 > narg) error->all(FLERR, "Illegal compute {} command", style);
      gridtypeflagl = utils::inumeric(FLERR, arg[iarg + 1], false, lmp);
      iarg += 2;
    }
  }

  delete acefimpl->basis_set;
  acefimpl->basis_set = new ACECTildeBasisSet(potential_file_name);
  cutmax = acefimpl->basis_set->cutoffmax;
  nelements = acefimpl->basis_set->nelements;
  memory->create(map,nelements+1,"python/acegridforce:map");
  for (int ik = 1; ik <= nelements+1; ik++) {
    for(int mu = 0; mu < nelements; mu++){
      if (mu != -1) {
        if (mu == ik - 1) {
          map[ik] = mu;
          //acefimpl->ace->element_type_mapping(ik) = mu;
        }
      }
    }
  }

  //# of rank 1, rank > 1 functions
  int ielem = 0;
  if (gridtypeflagl){
    ielem = nelements-1;
  }
  int n_r1, n_rp = 0;
  n_r1 = acefimpl->basis_set->total_basis_size_rank1[ielem];
  n_rp = acefimpl->basis_set->total_basis_size[ielem];

  int ncoeff = n_r1 + n_rp;

  //ndesc_base = 6;
  ndesc_base = 0;
  nvalues = ncoeff;
  ndesc = ncoeff;

  set_grid_global();
  set_grid_local();
  allocate_global();
  size_local_array_rows = ngridlocal + base_array_rows;

  // get Python function
  PyUtils::GIL lock;

  PyObject *pyMain = PyImport_AddModule("__main__");

  if (!pyMain) {
    PyUtils::Print_Errors();
    error->all(FLERR,"Could not initialize embedded Python");
  }

  char *fname = arg[5];
  pFunc = PyObject_GetAttrString(pyMain, fname);

  if (!pFunc) {
    PyUtils::Print_Errors();
    error->all(FLERR,"Could not find Python function");
  }

  lmpPtr = PY_VOID_POINTER(lmp);
}

/* ---------------------------------------------------------------------- */

FixPythonAceGridForce::~FixPythonAceGridForce()
{
  PyUtils::GIL lock;
  Py_CLEAR(lmpPtr);
  delete acefimpl;
  
  deallocate_py_beta();
  deallocate_grid();
  deallocate_short();
  deallocate_global();
  memory->destroy(map);
}

/* ---------------------------------------------------------------------- */

int FixPythonAceGridForce::setmask()
{
  return selected_callback;
}


/* ----------------------------------------------------------------------
   convert global array indexes to box coords
------------------------------------------------------------------------- */

void FixPythonAceGridForce::grid2x(int ix, int iy, int iz, double *x)
{
  x[0] = ix*delx;
  x[1] = iy*dely;
  x[2] = iz*delz;

  if (triclinic) domain->lamda2x(x, x);
}

void FixPythonAceGridForce::grid2xglobal(int igrid, double *x)
{
  int iz = igrid / (nx * ny);
  igrid -= iz * (nx * ny);
  int iy = igrid / nx;
  igrid -= iy * nx;
  int ix = igrid;

  x[0] = ix * delx;
  x[1] = iy * dely;
  x[2] = iz * delz;

  if (triclinic) domain->lamda2x(x, x);
  //printf(">>>>> ComputeGrid::grid2x\n");
}

/* ----------------------------------------------------------------------
   create arrays
------------------------------------------------------------------------- */

void FixPythonAceGridForce::allocate_grid()
{
  if (gridlocal_allocated){
    deallocate_grid();
  }
  if (nxlo <= nxhi && nylo <= nyhi && nzlo <= nzhi) {
    gridlocal_allocated = 1;
    memory->create4d_offset(gridlocal,ndesc,nzlo,nzhi,nylo,nyhi,
                            nxlo,nxhi,"python/acegridforce:gridlocal");
    memory->create(alocal, ngridlocal, ndesc, "python/acegridforce:alocal");
  }
}


void FixPythonAceGridForce::allocate_py_beta()
{
  if (allocated_py_beta){
    deallocate_py_beta();
  }
  memory->create(py_beta, size_local_array_rows, ndesc-ndesc_base, "python/acegridforce:py_beta");
  memory->create(py_beta_contig, size_local_array_rows*(ndesc-ndesc_base), "python/acegridforce:py_beta_contig");
  //memory->create(py_beta, size_global_array_rows, ndesc-ndesc_base, "python/acegridforce:py_beta");
  //memory->create(py_beta_contig, size_global_array_rows*(ndesc-ndesc_base), "python/acegridforce:py_beta_contig");
  allocated_py_beta = 1;
}

void FixPythonAceGridForce::allocate_global()
{
  if (allocated_global){
    deallocate_global();
  }
  //TODO - redo global vs local gridpoints to take betas of shape (ngridlocal,ncoeff) from python
  memory->create(e_grid, ngridlocal, "python/acegridforce:e_grid");
  memory->create(e_grid_all, ngridlocal, "python/acegridforce:e_grid_all");
  memory->create(e_grid_global, ngridglobal, "python/acegridforce:e_grid_global");
  //memory->create(e_grid, ngridglobal, "python/acegridforce:e_grid");
  //memory->create(e_grid_all, ngridglobal, "python/acegridforce:e_grid_all");
  allocated_global = 1;

}

/* ----------------------------------------------------------------------
   free arrays
------------------------------------------------------------------------- */

void FixPythonAceGridForce::deallocate_grid()
{
  if (gridlocal_allocated) {
    gridlocal_allocated = 0;
    memory->destroy4d_offset(gridlocal,nzlo,nylo,nxlo);
    memory->destroy(alocal);
  }
}

void FixPythonAceGridForce::deallocate_global()
{
  if (allocated_global){
    memory->destroy(e_grid);
    memory->destroy(e_grid_all);
    memory->destroy(e_grid_global);
  }
  allocated_global = 0;
}

void FixPythonAceGridForce::deallocate_py_beta()
{
  if (allocated_py_beta){
    memory->destroy(py_beta);
    memory->destroy(py_beta_contig);
  }
  allocated_py_beta = 0;
}

void FixPythonAceGridForce::deallocate_short()
{
  if (short_allocated){
    memory->destroy(gridneigh);
    memory->destroy(gridinside);
    memory->destroy(gridj);
    memory->destroy(gridtype);
  }
  short_allocated=0;
}
/* ----------------------------------------------------------------------
   set global grid
------------------------------------------------------------------------- */

void FixPythonAceGridForce::set_grid_global()
{
  // calculate grid layout

  triclinic = domain->triclinic;

  if (triclinic == 0) {
    prd = domain->prd;
    boxlo = domain->boxlo;
    sublo = domain->sublo;
    subhi = domain->subhi;
  } else {
    prd = domain->prd_lamda;
    boxlo = domain->boxlo_lamda;
    sublo = domain->sublo_lamda;
    subhi = domain->subhi_lamda;
  }

  double xprd = prd[0];
  double yprd = prd[1];
  double zprd = prd[2];

  delxinv = nx/xprd;
  delyinv = ny/yprd;
  delzinv = nz/zprd;

  delx = 1.0/delxinv;
  dely = 1.0/delyinv;
  delz = 1.0/delzinv;
}

/* ----------------------------------------------------------------------
   set local subset of grid that I own
   n xyz lo/hi = 3d brick that I own (inclusive)
------------------------------------------------------------------------- */

void FixPythonAceGridForce::set_grid_local()
{
  // nx,ny,nz = extent of global grid
  // indices into the global grid range from 0 to N-1 in each dim
  // if grid point is inside my sub-domain I own it,
  //   this includes sub-domain lo boundary but excludes hi boundary
  // ixyz lo/hi = inclusive lo/hi bounds of global grid sub-brick I own
  // if proc owns no grid cells in a dim, then ilo > ihi
  // if 2 procs share a boundary a grid point is exactly on,
  //   the 2 equality if tests insure a consistent decision
  //   as to which proc owns it

  double xfraclo,xfrachi,yfraclo,yfrachi,zfraclo,zfrachi;

  if (comm->layout != Comm::LAYOUT_TILED) {
    xfraclo = comm->xsplit[comm->myloc[0]];
    xfrachi = comm->xsplit[comm->myloc[0]+1];
    yfraclo = comm->ysplit[comm->myloc[1]];
    yfrachi = comm->ysplit[comm->myloc[1]+1];
    zfraclo = comm->zsplit[comm->myloc[2]];
    zfrachi = comm->zsplit[comm->myloc[2]+1];
  } else {
    xfraclo = comm->mysplit[0][0];
    xfrachi = comm->mysplit[0][1];
    yfraclo = comm->mysplit[1][0];
    yfrachi = comm->mysplit[1][1];
    zfraclo = comm->mysplit[2][0];
    zfrachi = comm->mysplit[2][1];
  }

  nxlo = static_cast<int> (xfraclo * nx);
  if (1.0*nxlo != xfraclo*nx) nxlo++;
  nxhi = static_cast<int> (xfrachi * nx);
  if (1.0*nxhi == xfrachi*nx) nxhi--;

  nylo = static_cast<int> (yfraclo * ny);
  if (1.0*nylo != yfraclo*ny) nylo++;
  nyhi = static_cast<int> (yfrachi * ny);
  if (1.0*nyhi == yfrachi*ny) nyhi--;

  nzlo = static_cast<int> (zfraclo * nz);
  if (1.0*nzlo != zfraclo*nz) nzlo++;
  nzhi = static_cast<int> (zfrachi * nz);
  if (1.0*nzhi == zfrachi*nz) nzhi--;

  ngridlocal = (nxhi - nxlo + 1) * (nyhi - nylo + 1) * (nzhi - nzlo + 1);
}

/* ----------------------------------------------------------------------
   copy coords to local array
------------------------------------------------------------------------- */

void FixPythonAceGridForce::assign_coords()
{
  int igrid = 0;
  for (int iz = nzlo; iz <= nzhi; iz++)
    for (int iy = nylo; iy <= nyhi; iy++)
      for (int ix = nxlo; ix <= nxhi; ix++) {
        alocal[igrid][0] = ix;
        alocal[igrid][1] = iy;
        alocal[igrid][2] = iz;
        double xgrid[3];
        grid2x(ix, iy, iz, xgrid);
        alocal[igrid][3] = xgrid[0];
        alocal[igrid][4] = xgrid[1];
        alocal[igrid][5] = xgrid[2];
        igrid++;
      }
}

/* ----------------------------------------------------------------------
   copy the 4d gridlocal array values to the 2d local array
------------------------------------------------------------------------- */

void FixPythonAceGridForce::copy_gridlocal_to_local_array()
{
  int igrid = 0;
  for (int iz = nzlo; iz <= nzhi; iz++)
    for (int iy = nylo; iy <= nyhi; iy++)
      for (int ix = nxlo; ix <= nxhi; ix++) {
        for (int icol = ndesc_base; icol < ndesc; icol++)
          alocal[igrid][icol] = gridlocal[icol][iz][iy][ix];
        igrid++;
      }
}

/* ----------------------------------------------------------------------
   calculate beta
------------------------------------------------------------------------- */

// this is a proxy for a call to the energy model
// beta is dE/dB^i, the derivative of the total
// energy w.r.t. to descriptors of grid point i

void FixPythonAceGridForce::compute_beta()
{
  int igrid = 0;
  for (int iz = nzlo; iz <= nzhi; iz++)
    for (int iy = nylo; iy <= nyhi; iy++)
      for (int ix = nxlo; ix <= nxhi; ix++) {
        for (int icol = 0; icol < ndesc-ndesc_base; icol++)
          beta[igrid][icol] = BETA_CONST + BETA_IGRID*igrid + BETA_ICOL*icol;
        igrid++;
      }
}
/* ----------------------------------------------------------------------
   grow grid
------------------------------------------------------------------------- */

void FixPythonAceGridForce::grow_grid(int newnmax)
{
  //if (newnmax <= nmax) return;
  //allocate local grid
  nmax=newnmax;
  if (gridlocal_allocated) {
    memory->destroy4d_offset(gridlocal,nzlo,nylo,nxlo);
    memory->destroy(alocal);
  }
  memory->create4d_offset(gridlocal,ndesc,nzlo,nzhi,nylo,nyhi,
                        nxlo,nxhi,"python/acegridforce:gridlocal");
  memory->create(alocal, ngridlocal, ndesc, "python/acegridforce:alocal");
  if (short_allocated) {
    memory->destroy(gridneigh);
    memory->destroy(gridinside);
    memory->destroy(gridj);
    memory->destroy(gridtype);
  }
  memory->create(gridneigh, nmax, 3, "python/acegridforce:gridneigh");
  memory->create(gridinside, nmax, "python/acegridforce:gridinside");
  memory->create(gridj, nmax, "python/acegridforce:gridj");
  memory->create(gridtype, nmax, "python/acegridforce:gridtype");
  short_allocated = 1;
  //populate short lists with 0s
  for (int jz = 0; jz < nmax; jz++){
    gridneigh[jz][0] = 0.00000000;
    gridneigh[jz][1] = 0.00000000;
    gridneigh[jz][2] = 0.00000000;
    gridinside[jz] = 0;
    gridj[jz] = 0;
    gridtype[jz] = 0;
  }
  gridlocal_allocated = 1;
}

/* ----------------------------------------------------------------------
   calculate B
------------------------------------------------------------------------- */
void FixPythonAceGridForce::compute(int eflag, int vflag)
{
  double fij[3];

  // compute descriptors for each gridpoint

  double** const x = atom->x;
  double **f = atom->f;
  const int* const mask = atom->mask;
  int * const type = atom->type;
  const int ntotal = atom->nlocal + atom->nghost;

  //zero out forces
  for (int jk = 0; jk < ntotal; jk++){
    f[jk][0] =0.;
    f[jk][1] =0.;
    f[jk][2] =0.;
  }

  // zero energy grid arrays
  for (int ik=0; ik < ngridglobal; ik++){
    e_grid[ik] = 0.0;
    e_grid_all[ik] = 0.0;
  }

  // grow grid arrays to match possible # of grid neighbors (analogous to jnum)
  nmax = ntotal + 1;
  grow_grid(nmax);  

  // first generate fingerprint,
  int igrid = 0;
  for (int iz = nzlo; iz <= nzhi; iz++)
    for (int iy = nylo; iy <= nyhi; iy++)
      for (int ix = nxlo; ix <= nxhi; ix++) {
        const int igrid_global = iz * (nx * ny) + iy * nx + ix;
        double xgrid[3];
        //double xgridglobal[3];
        grid2x(ix, iy, iz, xgrid);
        const double xtmp = xgrid[0];
        const double ytmp = xgrid[1];
        const double ztmp = xgrid[2];
        const int igrid_g = iz * (nx * ny) + iy * nx + ix;
        //grid2xglobal(igrid_g,xgridglobal);
        
        int ielem = 0;
        if (gridtypeflagl){
          ielem = nelements-1;
        }
        const int itype = ielem+1;
        delete acefimpl->ace;
        acefimpl->ace = new ACECTildeEvaluator(*acefimpl->basis_set);
        acefimpl->ace->compute_projections = true;
        acefimpl->ace->compute_b_grad = true;

        // leave these here b/c in the future, we may be able to allow
        // for different #s of descriptors per atom or grid type
        int n_r1, n_rp = 0;
        n_r1 = acefimpl->basis_set->total_basis_size_rank1[ielem];
        n_rp = acefimpl->basis_set->total_basis_size[ielem];

        //ACE element mapping
        acefimpl->ace->element_type_mapping.init(nelements+1);
        for (int ik = 1; ik <= nelements+1; ik++) {
          for(int mu = 0; mu < nelements; mu++){
            if (mu != -1) {
              if (mu == ik - 1) {
                map[ik] = mu;
                acefimpl->ace->element_type_mapping(ik) = mu;
              }
            }
          }
        }

        int ninside = 0;
        for (int j = 0; j < ntotal; j++) {
          if (!(mask[j] & groupbit)) continue;
          const double delxlg = xtmp - x[j][0];
          const double delylg = ytmp - x[j][1];
          const double delzlg = ztmp - x[j][2];
          const double rsq = delxlg * delxlg + delylg * delylg + delzlg * delzlg;
          int jtype = type[j];
          int jelem = map[jtype];
          double thiscut = acefimpl->basis_set->radial_functions->cut(ielem,jelem);
          const double rnorm = sqrt(rsq);
          if (rnorm <= (thiscut)  && rnorm > 1.e-20) {
            gridneigh[ninside][0] = x[j][0];
            gridneigh[ninside][1] = x[j][1];
            gridneigh[ninside][2] = x[j][2];
            gridinside[ninside] = ninside;
            gridj[ninside] = j;
            gridtype[ninside] = jtype;
            ninside++;
          }
        }
        //add in grid site at final index
        int ninsidegrid = ninside+1;
        int copy_igrid = igrid;
        const int baseind = 0;

        gridinside[ninside]=ninside;
        gridneigh[ninside][0] = xtmp; //good first
        gridneigh[ninside][1] = ytmp;
        gridneigh[ninside][2] = ztmp;
        gridtype[ninside]=itype;
        // perform ACE evaluation with short neighbor list
        acefimpl->ace->resize_neighbours_cache(ninside);
        acefimpl->ace->compute_atom(ninside, gridneigh, gridtype, ninside, gridinside);
        Array1D<DOUBLE_TYPE> Bs = acefimpl->ace->projections;
        // Accumulate descriptors on grid array
        //   Accumulate energy (linear model for debugging)
        for (int icoeff = 0; icoeff < ndesc; icoeff++){
          //alocal[igrid][ndesc_base + icoeff] += Bs(icoeff);
          //e_grid[igrid_global] += Bs(icoeff)*py_beta[0][icoeff];
          gridlocal[ndesc_base + icoeff][iz][iy][ix] = Bs(icoeff);
#if ENERGY_ROW
          e_grid[igrid_global] += Bs(icoeff)*py_beta[0][icoeff];
#endif
        }
        //Accumulate forces
        // sum over neighbors jj
        // sum over descriptor indices k=iicoeff
        // multiply dE_I/dB_I * dB_I^k/drj and add to atom->f 
        for (int jj =0; jj < ninside;jj++){
          int mj = gridj[jj];
          for (int iicoeff = 0; iicoeff < ndesc; iicoeff++){
            DOUBLE_TYPE fx_dB = acefimpl->ace->neighbours_dB(iicoeff,jj,0);
            DOUBLE_TYPE fy_dB = acefimpl->ace->neighbours_dB(iicoeff,jj,1);
            DOUBLE_TYPE fz_dB = acefimpl->ace->neighbours_dB(iicoeff,jj,2);
            f[atom->tag[mj]-1][0] -= fx_dB *py_beta[igrid_global+base_array_rows][iicoeff];
            f[atom->tag[mj]-1][1] -= fy_dB *py_beta[igrid_global+base_array_rows][iicoeff];
            f[atom->tag[mj]-1][2] -= fz_dB *py_beta[igrid_global+base_array_rows][iicoeff];
          }
        }
        
        igrid++;
      }
}

/* ----------------------------------------------------------------------
   compute energy scalar for fix energy
------------------------------------------------------------------------- */

double FixPythonAceGridForce::compute_scalar()
{
#if MPI_NUMPY
  MPI_Allreduce(e_grid, e_grid_all, ngridglobal, MPI_DOUBLE, MPI_SUM, world);
  double etot = 0.0;
  for (int kk = 0; kk < ngridglobal; kk++){
    etot += e_grid_all[kk];
  }
#endif
#if !MPI_NUMPY
  double etot = 0.0;
  for (int kk = 0; kk < ngridglobal; kk++){
    etot += e_grid[kk];
  }
#endif
  return etot;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void FixPythonAceGridForce::base_allocate()
{
  base_allocated = 1;
  int n = atom->ntypes;

  map = new int[n+1];
}

/* ---------------------------------------------------------------------- */

void FixPythonAceGridForce::end_of_step()
{
  PyUtils::GIL lock;

  PyObject * result = PyObject_CallFunction((PyObject*)pFunc, (char *)"O", (PyObject*)lmpPtr);

  if (!result) {
    PyUtils::Print_Errors();
    error->all(FLERR,"Fix python/acegridforce end_of_step() method failed");
  }

  Py_CLEAR(result);
}

void FixPythonAceGridForce::process_pyarr(PyObject* arr)
{
  int prank = 0;
#if MPI_NUMPY
  MPI_Comm_rank(MPI_COMM_WORLD, &prank);
#endif
  if (prank == 0 ){
 
    //TODO - finish version that can take local dEdB from python with shape (ngridlocal,ncoeff) 
    double* pybeta = (double*)PyArray_DATA(arr);
    deallocate_py_beta();
    allocate_py_beta();
    int pyrows = size_global_array_rows;
    int pycols = ndesc-ndesc_base;
#if MPI_NUMPY
    MPI_Allreduce(pybeta, py_beta_contig, (size_global_array_rows*(ndesc-ndesc_base)), MPI_DOUBLE, MPI_SUM, world);
#endif
    for (int pyi =0; pyi < pyrows; pyi++){
      for (int pyj =0; pyj < pycols; pyj++){
#ifdef DEBUG_NUMPY_INTERFACE
        printf("pybeta[%d][%d] = %f \n" , pyi,pyj, pybeta[pyi*pycols + pyj]);
#endif
#if MPI_NUMPY
        double resulti = py_beta_contig[pyi*pycols + pyj];
#endif
#if !MPI_NUMPY
        double resulti = pybeta[pyi*pycols + pyj];
#endif
        py_beta[pyi][pyj] = resulti;
      }
    }
  }

}
/* ---------------------------------------------------------------------- */

void FixPythonAceGridForce::post_force(int vflag)
{
  if (update->ntimestep % nevery != 0) return;

  PyUtils::GIL lock;
  char fmt[] = "Oi";

  PyObject * result = PyObject_CallFunction((PyObject*)pFunc, fmt, (PyObject*)lmpPtr, vflag);

  if (!result) {
    PyUtils::Print_Errors();
    error->all(FLERR,"Fix python/acegridforce post_force() method failed");
  }

  Py_CLEAR(result);
}

/* ---------------------------------------------------------------------- */

void FixPythonAceGridForce::pre_force(int vflag)
{
  //if (update->ntimestep % nevery != 0) return;

  PyUtils::GIL lock;


  char fmt[] = "O";

  PyObject * result = PyObject_CallFunction((PyObject*)pFunc, fmt, (PyObject*)lmpPtr, vflag);

  deallocate_grid();
  allocate_grid();

  // directly process python beta array
  process_pyarr(result);
  // compute ace descriptors on the grid
  compute(1,0);

  if (!result) {
    PyUtils::Print_Errors();
    error->all(FLERR,"Fix python/acegridforce pre_force() method failed");
  }

  Py_CLEAR(result);
}
