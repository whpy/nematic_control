#ifndef FLDOP_H_
#define FLDOP_H_
 
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
#include "cuComplexBinOp.h"
// #include "cudaErr.h"
struct Mesh;
struct field;
struct param;

inline void FwdTrans(Mesh* pmesh, float* phys, cuComplex* spec);
inline void BwdTrans(Mesh* pmesh, cuComplex* spec, float* phys);
__global__ void coeff(float *f, int Nx, int Ny);

// __Device: phys field multiplication: pc(x) = C*pa(x,y)*pb(x,y). prefix p denotes physical
__global__ void FldMul(float* pa, float* pb, float C, float* pc, int Nx, int Ny);

// __Device: phys field addition: pc(x,y) = a*pa(x,y) + b*pb(x,y). prefix p denotes physical
__global__  void FldAdd(float* pa, float* pb, float a, float b, float* pc, int Nx, int Ny);
__global__  void FldAdd(float a, float* pa, float b, float* pb, float* pc, int Nx, int Ny);

// divide a factor after transforming from spectral to physical
__global__ void coeff(float *f, int Nx, int Ny);

__global__
// set physical field equals to a constant field
void FldSet(float * pf, float c, int Nx, int Ny);

__global__
// set two physical field equals
void FldSet(float * pf, float* c, int Nx, int Ny);

__global__
void SpecSet(cuComplex * pa, cuComplex* pb, int Nxh, int Ny);

__global__ 
// spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
void SpecAdd(cuComplex* spa, cuComplex* spb, float a, float b, cuComplex* spc, int Nxh, int Ny);

__global__
void SpecAdd(float a, cuComplex* spa, float b, cuComplex* spb, cuComplex* spc, int Nxh, int Ny);

__global__ void xDerivD(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny);
inline void xDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh);

inline void yDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh);
__global__ void yDerivD(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny);

__global__ void reality_func(cuComplex *spec, int Nxh, int Ny);

#endif