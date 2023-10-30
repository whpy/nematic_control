#ifndef FLDOP_H_
#define FLDOP_H_
 
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
#include <Basic/cuComplexBinOp.hpp>
#include <Basic/cudaErr.h>

#include <Basic/Field.h>
#include <Basic/Mesh.h>

/*******************************************************************
General functions
*********************************************************************/

// generate the 0-1 sequence to consider which frequency to be deprecated
__global__
void cutoff_func(Qreal* cutoff, int Nxh, int Ny, int BSZ);
// deprecate the high frequencies determined by the cutoff array
__global__
void dealiasing_func(Qcomp* f_spec, Qreal* cutoff,int Nxh, int Ny, int BSZ);
// in the referenced code, this function occurs more frequently than the dealiasing,
// it is applied after each time the nonlinear function is called. so maybe it is the
// main reason to retain the numerical precision.
__global__
void symmetry_func(Qcomp f_spec[], int Nxh, int Ny, int BSZ);

inline void FwdTrans(Mesh* pmesh, Qreal* phys, Qcomp* spec);

inline void BwdTrans(Mesh* pmesh, Qcomp* spec, Qreal* phys);

// __Device: pc = a*pa + b*pb
__global__  
void FldAdd(Qreal a, Qreal* pa, Qreal b, Qreal* pb, Qreal* pc, int Nx, int Ny, int BSZ);
// __Device: pc = a*pa + b
__global__
void FldAdd(Qreal a, Qreal* pa, Qreal b, Qreal* pc, int Nx, int Ny, int BSZ);


// divide a factor after transforming from spectral to physical
__global__ 
void coeff(Qreal *f, int Nx, int Ny, int BSZ);

__global__
// set physical Field equals to a constant Field
void FldSet(Qreal * pf, Qreal c, int Nx, int Ny, int BSZ);

__global__
// set two physical Field equals
void FldSet(Qreal * pf, Qreal* c, int Nx, int Ny, int BSZ);

__global__
void SpecSet(Qcomp * pa, Qcomp* pb, int Nxh, int Ny, int BSZ);

__global__
void SpecSet(Qcomp * pa, Qcomp c, int Nxh, int Ny, int BSZ);

__global__ 
// spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
void SpecAdd(Qcomp* spa, Qcomp* spb, Qreal a, Qreal b, Qcomp* spc, int Nxh, int Ny, int BSZ);

__global__
void SpecAdd(Qreal a, Qcomp* spa, Qreal b, Qcomp* spb, Qcomp* spc, int Nxh, int Ny, int BSZ);

// spectral multiplication: spc(k,l) = C*spa(k,l)*spb(k,l)
__global__ 
void SpecMul(Qcomp* spa, Qcomp* spb, Qreal C, Qcomp*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(Qcomp* spa, Qreal C, Qcomp*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(Qcomp* spa, Qcomp C, Qcomp*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(Qcomp* spa, Qcomp* spb, Qcomp C, Qcomp*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(Qreal* spa, Qcomp* spb, Qreal C, Qcomp*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(Qcomp* spa, Qreal* spb, Qreal C, Qcomp*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(Qreal* spa, Qcomp* spb, Qcomp C, Qcomp*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(Qcomp* spa, Qreal* spb, Qcomp C, Qcomp*spc, int Nxh, int Ny, int BSZ);

__global__ 
void xDerivD(Qcomp *ft, Qcomp *dft, Qreal* kx, int Nxh, int Ny, int BSZ);
void xDeriv(Qcomp *ft, Qcomp *dft, Mesh *mesh);

__global__ 
void yDerivD(Qcomp *ft, Qcomp *dft, Qreal* ky, int Nxh, int Ny, int BSZ);
void yDeriv(Qcomp *ft, Qcomp *dft, Mesh *mesh);

// maintain the symmetry on y wave axis
__global__ 
void reality_func(Qcomp *spec, int Nxh, int Ny, int BSZ);

__global__ 
void laplacian_funcD(Qcomp *ft, Qcomp *lft, int Nxh, int Ny, Qreal* k_squared);
void laplacian_func(Qcomp *ft, Qcomp *lft, Mesh* mesh);

__global__ 
void vel_funcD(Qcomp* w_spec, Qcomp* u_spec, Qcomp* v_spec, 
                            Qreal* k_squared, Qreal* kx, Qreal*ky, int Nxh, int Ny);
void vel_func(Field w, Field u, Field v);

#endif
