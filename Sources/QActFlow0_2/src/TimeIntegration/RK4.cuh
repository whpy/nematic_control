#include <Basic/QActFlow.h>

// 4 steps of RK4 under spectral linear factor trick 



__global__
void integrate_func0(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new,
                    float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);

__global__  
void integrate_func1(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
                    float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);

__global__ 
void integrate_func2(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);

__global__ 
void integrate_func3(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);

__global__ 
void integrate_func4(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);