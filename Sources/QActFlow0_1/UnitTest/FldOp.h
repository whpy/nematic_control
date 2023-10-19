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

/*******************************************************************
General functions
*********************************************************************/
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
void SpecSet(cuComplex * pa, cuComplex c, int Nxh, int Ny);

__global__ 
// spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
void SpecAdd(cuComplex* spa, cuComplex* spb, float a, float b, cuComplex* spc, int Nxh, int Ny);

__global__
void SpecAdd(float a, cuComplex* spa, float b, cuComplex* spb, cuComplex* spc, int Nxh, int Ny);

__global__ 
void xDerivD(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny);
inline void xDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh);

__global__ 
void yDerivD(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny);
inline void yDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh);

// maintain the symmetry on y wave axis
__global__ 
void reality_func(cuComplex *spec, int Nxh, int Ny);

__global__ 
void laplacian_funcD(cuComplex *ft, cuComplex *lft, int Nxh, int Ny, float* k_squared);
inline void laplacian_func(cuComplex *ft, cuComplex *lft, Mesh* mesh);

__global__ 
void vel_funcD(cuComplex* w_spec, cuComplex* u_spec, cuComplex* v_spec, 
                            float* k_squared, float* kx, float*ky, int Nxh, int Ny);
inline void vel_func(field w, field u, field v);

// 4 steps of RK4 under spectral linear factor trick 
__global__
void integrate_func0(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new,
                    float* IF, float* IFh, int Nxh, int Ny, float dt);

__global__  
void integrate_func1(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
                    float* IF, float* IFh, int Nxh, int Ny, float dt);

__global__ 
void integrate_func2(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt);

__global__ 
void integrate_func3(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt);

__global__ 
void integrate_func4(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt);

/*******************************************************************
specialized functions
*********************************************************************/
__global__ 
void S_func(float* r1, float*r2, float* S, int Nx, int Ny);

void curr_func(field r1curr, field r2curr, field wcurr, field u, field v, field S);

inline void r1nonl_func(field r1nonl, field r1nonl_appr, field r1, field r2, field w, 
                        field u, field v, field S, float lambda, float cn, float Pe);

inline void r2nonl_func(field r2nonl, field r2nonl_appr, field r1, field r2, field w, 
                        field u, field v, field S, float lambda, float cn, float Pe);

inline void wnonl_func(field wnonl, field wnonl_appr, field appr1, field p11, field p12, field p21, field r1, field r2, field w, 
                        field u, field v, field alpha, field S, float Re, float Er, float cn, float lambda);

inline void pCross_func(field p,field appr, field r1, field r2);

inline void pSingle_func(field p, field appr, field r, field S, field alpha, float lambda, float cn);

inline void p11nonl_func(field p11, field appr, field appr1, field r1, field r2, field S, 
                        field alpha, float lambda, float cn);

inline void p12nonl_func(field p12, field appr, field appr1, field r1, field r2, field S, 
                        field alpha, float lambda, float cn);

inline void p21nonl_func(field p21, field appr, field appr1, field r1, field r2, field S, 
                        field alpha, float lambda, float cn);

__global__
void precompute_funcD(field r1, field r2, field w, field alpha, int Nx, int Ny);

__global__
void r1lin_func(float* IFr1h, float* IFr1, float* k_squared, float Pe, float cn, float dt);

__global__
void r2lin_func(float* IFr2h, float* IFr2, float* k_squared, float Pe, float cn, float dt);

__global__
void wlin_func(float* IFwh, float* IFw, float* k_squared, float Re, float cf, float dt);


#endif
