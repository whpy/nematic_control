#ifndef __STREAMFUNC_CUH
#define __STREAMFUNC_CUH

#include <Basic/QActFlow.h>
#include <Field/Field.h>
#include <Basic/FldOp.cuh>

__global__ 
void vel_funcD(cuComplex* w_spec, cuComplex* u_spec, cuComplex* v_spec, 
                            float* k_squared, float* kx, float*ky, int Nxh, int Ny);
void vel_func(Field w, Field u, Field v);

__global__
void r1lin_func(float* IFr1h, float* IFr1, float* k_squared, float Pe, float cn, float dt, int Nxh, int Ny, int BSZ);

__global__
void r2lin_func(float* IFr2h, float* IFr2, float* k_squared, float Pe, float cn, float dt, int Nxh, int Ny, int BSZ);

__global__
void wlin_func(float* IFwh, float* IFw, float* k_squared, float Re, float cf, float dt, int Nxh, int Ny, int BSZ);

__global__ 
void S_func(float* r1, float*r2, float* S, int Nx, int Ny);

void curr_func(Field r1curr, Field r2curr, Field wcurr, Field u, Field v, Field S);

void r1nonl_func(Field r1nonl, Field r1nonl_appr, Field r1, Field r2, Field w, 
                        Field u, Field v, Field S, float lambda, float cn, float Pe);

void r2nonl_func(Field r2nonl, Field r2nonl_appr, Field r1, Field r2, Field w, 
                        Field u, Field v, Field S, float lambda, float cn, float Pe);

inline void wnonl_func(Field wnonl, Field wnonl_appr, Field appr1, Field p11, Field p12, Field p21, Field r1, Field r2, Field w, 
                        Field u, Field v, Field alpha, Field S, float Re, float Er, float cn, float lambda);

inline void pCross_func(Field p,Field appr, Field r1, Field r2);

inline void pSingle_func(Field p, Field appr, Field r, Field S, Field alpha, float lambda, float cn);

inline void p11nonl_func(Field p11, Field appr, Field appr1, Field r1, Field r2, Field S, 
                        Field alpha, float lambda, float cn);

inline void p12nonl_func(Field p12, Field appr, Field appr1, Field r1, Field r2, Field S, 
                        Field alpha, float lambda, float cn);

inline void p21nonl_func(Field p21, Field appr, Field appr1, Field r1, Field r2, Field S, 
                        Field alpha, float lambda, float cn);

__global__
void precompute_funcD(Field r1, Field r2, Field w, Field alpha, int Nx, int Ny);

#endif // end of __STREAMFUNC_CUH