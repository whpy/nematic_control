#ifndef __RK4_H 
#define __RK4_H

#include <Basic/QActFlow.h>
#include <Basic/Mesh.h>
#include <Field/Field.h>

// 4 steps of RK4 under spectral linear factor trick 



__global__
void integrate_func0D(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new,
                    float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);
void integrate_func0(Field* f_old, Field* f_curr, Field* f_new, float* IF, float* IFh, float dt);

__global__  
void integrate_func1D(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
                    float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);
void integrate_func1(Field* f_old, Field* f_curr, Field* f_new, Field*f_nonl, float* IF, float* IFh, float dt);

__global__ 
void integrate_func2D(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);
void integrate_func2(Field* f_old, Field* f_curr, Field* f_new, Field*f_nonl, float* IF, float* IFh, float dt);

__global__ 
void integrate_func3D(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);
void integrate_func3(Field* f_old, Field* f_curr, Field* f_new, Field*f_nonl, float* IF, float* IFh, float dt);

__global__ 
void integrate_func4D(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt);
void integrate_func4(Field* f_old, Field* f_curr, Field* f_new, Field*f_nonl, float* IF, float* IFh, float dt);

#endif // end of RK4.h