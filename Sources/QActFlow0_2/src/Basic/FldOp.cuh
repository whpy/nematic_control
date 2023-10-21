#ifndef FLDOP_H_
#define FLDOP_H_
 
#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
#include <Basic/cuComplexBinOp.h>
#include <Basic/cudaErr.h>

#include <Field/Field.h>
#include <Basic/Mesh.h>

/*******************************************************************
General functions
*********************************************************************/
void FwdTrans(Mesh* pmesh, float* phys, cuComplex* spec);
void BwdTrans(Mesh* pmesh, cuComplex* spec, float* phys);
__global__ void coeff(float *f, int Nx, int Ny, int BSZ);

// __Device: phys Field multiplication: pc(x) = C*pa(x,y)*pb(x,y). prefix p denotes physical
__global__ void FldMul(float* pa, float* pb, float C, float* pc, int Nx, int Ny, int BSZ);

// __Device: phys Field addition: pc(x,y) = a*pa(x,y) + b*pb(x,y). prefix p denotes physical
__global__  void FldAdd(float* pa, float* pb, float a, float b, float* pc, int Nx, int Ny, int BSZ);
// __Device: pc = a*pa + b*pb
__global__  void FldAdd(float a, float* pa, float b, float* pb, float* pc, int Nx, int Ny, int BSZ);
// __Device: pc = a*pa + b
__global__
void FldAdd(float a, float* pa, float b, float* pc, int Nx, int Ny, int BSZ);


// divide a factor after transforming from spectral to physical
__global__ void coeff(float *f, int Nx, int Ny, int BSZ);

__global__
// set physical Field equals to a constant Field
void FldSet(float * pf, float c, int Nx, int Ny, int BSZ);

__global__
// set two physical Field equals
void FldSet(float * pf, float* c, int Nx, int Ny, int BSZ);

__global__
void SpecSet(cuComplex * pa, cuComplex* pb, int Nxh, int Ny, int BSZ);

__global__
void SpecSet(cuComplex * pa, cuComplex c, int Nxh, int Ny, int BSZ);

__global__ 
// spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
void SpecAdd(cuComplex* spa, cuComplex* spb, float a, float b, cuComplex* spc, int Nxh, int Ny, int BSZ);

__global__
void SpecAdd(float a, cuComplex* spa, float b, cuComplex* spb, cuComplex* spc, int Nxh, int Ny, int BSZ);

// spectral multiplication: spc(k,l) = C*spa(k,l)*spb(k,l)
__global__ 
void SpecMul(cuComplex* spa, cuComplex* spb, float C, cuComplex*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(cuComplex* spa, cuComplex* spb, cuComplex C, cuComplex*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(float* spa, cuComplex* spb, float C, cuComplex*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(cuComplex* spa, float* spb, float C, cuComplex*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(float* spa, cuComplex* spb, cuComplex C, cuComplex*spc, int Nxh, int Ny, int BSZ);
__global__ 
void SpecMul(cuComplex* spa, float* spb, cuComplex C, cuComplex*spc, int Nxh, int Ny, int BSZ);

__global__ 
void xDerivD(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny, int BSZ);
void xDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh);

__global__ 
void yDerivD(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny, int BSZ);
void yDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh);

// maintain the symmetry on y wave axis
__global__ 
void reality_func(cuComplex *spec, int Nxh, int Ny, int BSZ);

__global__ 
void laplacian_funcD(cuComplex *ft, cuComplex *lft, int Nxh, int Ny, float* k_squared);
void laplacian_func(cuComplex *ft, cuComplex *lft, Mesh* mesh);

__global__ 
void vel_funcD(cuComplex* w_spec, cuComplex* u_spec, cuComplex* v_spec, 
                            float* k_squared, float* kx, float*ky, int Nxh, int Ny);
void vel_func(Field w, Field u, Field v);

// 4 steps of RK4 under spectral linear factor trick 
// __global__
// void integrate_func0(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new,
//                     float* IF, float* IFh, int Nxh, int Ny, float dt);

// __global__  
// void integrate_func1(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
//                     float* IF, float* IFh, int Nxh, int Ny, float dt);

// __global__ 
// void integrate_func2(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
//                         cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt);

// __global__ 
// void integrate_func3(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
//                         cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt);

// __global__ 
// void integrate_func4(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
//                         cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt);

/*******************************************************************
specialized functions
*********************************************************************/
// __global__ 
// void S_func(float* r1, float*r2, float* S, int Nx, int Ny);

// void curr_func(Field r1curr, Field r2curr, Field wcurr, Field u, Field v, Field S);

// void r1nonl_func(Field r1nonl, Field r1nonl_appr, Field r1, Field r2, Field w, 
//                         Field u, Field v, Field S, float lambda, float cn, float Pe);

// void r2nonl_func(Field r2nonl, Field r2nonl_appr, Field r1, Field r2, Field w, 
//                         Field u, Field v, Field S, float lambda, float cn, float Pe);

// inline void wnonl_func(Field wnonl, Field wnonl_appr, Field appr1, Field p11, Field p12, Field p21, Field r1, Field r2, Field w, 
//                         Field u, Field v, Field alpha, Field S, float Re, float Er, float cn, float lambda);

// inline void pCross_func(Field p,Field appr, Field r1, Field r2);

// inline void pSingle_func(Field p, Field appr, Field r, Field S, Field alpha, float lambda, float cn);

// inline void p11nonl_func(Field p11, Field appr, Field appr1, Field r1, Field r2, Field S, 
//                         Field alpha, float lambda, float cn);

// inline void p12nonl_func(Field p12, Field appr, Field appr1, Field r1, Field r2, Field S, 
//                         Field alpha, float lambda, float cn);

// inline void p21nonl_func(Field p21, Field appr, Field appr1, Field r1, Field r2, Field S, 
//                         Field alpha, float lambda, float cn);

// __global__
// void precompute_funcD(Field r1, Field r2, Field w, Field alpha, int Nx, int Ny);

// __global__
// void r1lin_func(float* IFr1h, float* IFr1, float* k_squared, float Pe, float cn, float dt, int Nxh, int Ny);

// __global__
// void r2lin_func(float* IFr2h, float* IFr2, float* k_squared, float Pe, float cn, float dt, int Nxh, int Ny);

// __global__
// void wlin_func(float* IFwh, float* IFw, float* k_squared, float Re, float cf, float dt, int Nxh, int Ny);

/******************************************************************************* 
 * operator functions                                                          * 
 *******************************************************************************/
// operators for field solvers
// divide a factor after transforming from spectral to physical
// __global__ void coeff(float *f, int Nx, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nx + i;
//     if(i<Nx && j<Ny){
//         f[index] = f[index]/(Nx*Ny);
//     }
// }

// //update the spectral space based on the value in physical space
// inline void FwdTrans(Mesh* pmesh, float* phys, cuComplex* spec){
//     cufft_error_func( cufftExecR2C(pmesh->transf, phys, spec));
//     cuda_error_func( cudaDeviceSynchronize() );
// }

// //update the physics space based on the value in spectral space
// inline void BwdTrans(Mesh* pmesh, cuComplex* spec, float* phys){
//     int Nx = pmesh->Nx;
//     int Ny = pmesh->Ny;
//     int BSZ = pmesh->BSZ;
//     cufft_error_func( cufftExecC2R(pmesh->inv_transf, spec, phys));
//     cuda_error_func( cudaDeviceSynchronize() );
//     dim3 dimGrid = pmesh->dimGridp;
//     dim3 dimBlock = pmesh->dimBlockp;
//     coeff<<<dimGrid, dimBlock>>>(phys, Nx, Ny, BSZ);
//     // in the referenced source code, they seem a little bit abuse synchronization, this
//     // may be a point that we could boost the performance in the future. we temporarily
//     // comply with the same principle that our code would at least perform no worse than theirs
//     cuda_error_func( cudaPeekAtLastError() );
// 	cuda_error_func( cudaDeviceSynchronize() );
// }

// __global__
// // __Device: phys field multiplication: pc(x) = C*pa(x,y)*pb(x,y). prefix p denotes physical
// void FldMul(float* pa, float* pb, float C, float* pc, int Nx, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nx + i;
//     if(i<Nx && j<Ny){
//         pc[index] = C*pa[index]*pb[index];
//     }
// }

// __global__ 
// // __Device: phys field addition: pc(x,y) = a*pa(x,y) + b*pb(x,y). prefix p denotes physical
// void FldAdd(float* pa, float* pb, float a, float b, float* pc, int Nx, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nx + i;
//     if(i<Nx && j<Ny){
//         pc[index] = a*pa[index] + b*pb[index];
//     }
// }

// __global__
// void FldAdd(float a, float* pa, float b, float* pb, float* pc, int Nx, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nx + i;
//     if(i<Nx && j<Ny){
//         pc[index] = a*pa[index] + b*pb[index];
//     }
// }
// __global__
// // set physical field equals to a constant field
// void FldSet(float * pf, float c, int Nx, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nx + i;
//     if (i<Nx && j<Ny){
//         pf[index] = c;
//     }
// }
// __global__
// // set two physical field equals
// void FldSet(float * pf, float* c, int Nx, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nx + i;
//     if (i<Nx && j<Ny){
//         pf[index] = c[index];
//     }
// }
// __global__
// void SpecSet(cuComplex * pa, cuComplex* pb, int Nxh, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     if (i<Nxh && j<Ny){
//         pa[index] = pb[index];
//     }
// }

// __global__
// void SpecSet(cuComplex * pa, cuComplex c, int Nxh, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     if (i<Nxh && j<Ny){
//         pa[index] = c;
//     }
// }

// __global__ 
// // spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
// void SpecAdd(cuComplex* spa, cuComplex* spb, float a, float b, cuComplex* spc, int Nxh, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     if(i<Nxh && j<Ny){
//         spc[index] = a* spa[index] + b* spb[index];
//     }
// }

// __global__
// void SpecAdd(float a, cuComplex* spa, float b, cuComplex* spb, cuComplex* spc, int Nxh, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     if(i<Nxh && j<Ny){
//         spc[index] = a* spa[index] + b* spb[index];
//     }
// }

// __global__ void xDerivD(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     if(i<Nxh && j<Ny){
//         dft[index] = ft[index]*im()*kx[i];
//     }
// }
// inline void xDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh){
//     dim3 dimGrid = mesh->dimGridp;
//     dim3 dimBlock = mesh->dimBlockp;
//     xDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->kx, mesh->Nxh, mesh->Ny, mesh->BSZ);
//     cuda_error_func( cudaPeekAtLastError() );
// 	cuda_error_func( cudaDeviceSynchronize() );
// }

// __global__ void yDerivD(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny, int BSZ){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     if(i<Nxh && j<Ny){
//         dft[index] = ft[index]*im()*ky[j];
//     }
// }
// inline void yDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh){
//     dim3 dimGrid = mesh->dimGridp;
//     dim3 dimBlock = mesh->dimBlockp;
//     yDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->ky,mesh->Nxh, mesh->Ny, mesh->BSZ);
//     cuda_error_func( cudaPeekAtLastError() );
// 	cuda_error_func( cudaDeviceSynchronize() );
// }

#endif
