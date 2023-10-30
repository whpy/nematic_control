#ifndef __STREAMFUNC_CUH
#define __STREAMFUNC_CUH

#include <Basic/QActFlow.h>
#include <Basic/Field.h>
#include <Basic/FldOp.cuh>

__global__ 
void vel_funcD(Qcomp* w_spec, Qcomp* u_spec, Qcomp* v_spec, 
                            Qreal* k_squared, Qreal* kx, Qreal*ky, int Nxh, int Ny);

void vel_func(Field *w, Field *u, Field *v);

__global__
void r1lin_func(Qreal* IFr1h, Qreal* IFr1, Qreal* k_squared, Qreal Pe, Qreal cn, Qreal dt, int Nxh, int Ny, int BSZ);

__global__
void r2lin_func(Qreal* IFr2h, Qreal* IFr2, Qreal* k_squared, Qreal Pe, Qreal cn, Qreal dt, int Nxh, int Ny, int BSZ);

__global__
void wlin_func(Qreal* IFwh, Qreal* IFw, Qreal* k_squared, Qreal Re, Qreal cf, Qreal dt, int Nxh, int Ny, int BSZ);

__global__ 
void S_funcD(Qreal* r1, Qreal*r2, Qreal* S, int Nx, int Ny, int BSZ);
void S_func(Field* r1, Field* r2, Field* S);

void curr_func(Field *r1curr, Field *r2curr, Field *wcurr, Field *u, Field *v, Field *S);

void r1nonl_func(Field *r1nonl, Field *aux, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *S, Qreal lambda, Qreal cn, Qreal Pe);

void r2nonl_func(Field *r2nonl, Field *aux, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *S, Qreal lambda, Qreal cn, Qreal Pe);

void wnonl_func(Field *wnonl, Field *aux, Field *aux1, Field *p11, Field *p12, Field *p21, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *alpha, Field *S, Qreal Re, Qreal Er, Qreal cn, Qreal lambda);

void pCross_func(Field *p,Field *aux, Field *r1, Field *r2);

void pSingle_func(Field *p, Field *aux, Field *r, Field *S, Field *alpha, Qreal lambda, Qreal cn);

void p11nonl_func(Field *p11, Field *aux, Field *aux1, Field *r1, Field *r2, Field *S, 
                        Field *alpha, Qreal lambda, Qreal cn);

void p12nonl_func(Field *p12, Field *aux, Field *aux1, Field *r1, Field *r2, Field *S, 
                        Field *alpha, Qreal lambda, Qreal cn);

void p21nonl_func(Field *p21, Field *aux, Field *aux1, Field *r1, Field *r2, Field *S, 
                        Field *alpha, Qreal lambda, Qreal cn);


#endif // end of __STREAMFUNC_CUH