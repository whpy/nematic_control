#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>


// 4 steps of RK4 under spectral linear factor trick 

//RK4 integrating steps
// du/dt = \alpha*u + F(t,u)
// IFh = exp(\alpha*dt/2). IF = exp(\alpha*dt)
// u_{n+1} = u_{n}*IF + 1/6*(a*IF + 2b*IFh + 2c*IFh + d)
// a_n = dt*F(t_n,u_n)
// b_n = dt*F(t_n+dt/2, (u_n+a_n/2)*IFh)
// c_n = dt*F(t_n+dt/2, u_n*IFh + b_n/2)
// d_n = dt*F(t_n+dt, u_n*IF + c_n*IFh)

// preparation before RK4 integration
// prepare input ucurr(u_n) for computation of a_n
void integrate_func0(Field *u_old, Field *u_curr, Field *u_new, 
float *IF, float *IFh);


// compute the a_n
void integrate_func1(Field *u_old, Field *u_curr, Field *u_new, Field* u_nonl, float dt, 
float *IF, float *IFh);
__global__ 
void integrate_func1D(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
            float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);
// __global__  
// void integrate_func1(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
//                     float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);

__global__ 
void integrate_func2(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);

__global__ 
void integrate_func3(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);

__global__ 
void integrate_func4(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);