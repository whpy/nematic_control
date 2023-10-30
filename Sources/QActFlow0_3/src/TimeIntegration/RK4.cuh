#ifndef __RK4_H 
#define __RK4_H

#include <Basic/QActFlow.h>
#include <Basic/Mesh.h>
#include <Basic/Field.h>

// 4 steps of RK4 under spectral linear factor trick 



__global__
void integrate_func0D(Qcomp* spec_old, Qcomp* spec_curr, Qcomp* spec_new,
                    Qreal* IF, Qreal* IFh, int Nxh, int Ny, int BSZ, Qreal dt);
void integrate_func0(Field* f_old, Field* f_curr, Field* f_new, Qreal* IF, Qreal* IFh, Qreal dt);

__global__  
void integrate_func1D(Qcomp* spec_old, Qcomp* spec_curr, Qcomp* spec_new, Qcomp* spec_nonl,
                    Qreal* IF, Qreal* IFh, int Nxh, int Ny, int BSZ, Qreal dt);
void integrate_func1(Field* f_old, Field* f_curr, Field* f_new, Field*f_nonl, Qreal* IF, Qreal* IFh, Qreal dt);

__global__ 
void integrate_func2D(Qcomp* spec_old, Qcomp* spec_curr, Qcomp* spec_new, 
                        Qcomp* spec_nonl,Qreal* IF, Qreal* IFh, int Nxh, int Ny, int BSZ, Qreal dt);
void integrate_func2(Field* f_old, Field* f_curr, Field* f_new, Field*f_nonl, Qreal* IF, Qreal* IFh, Qreal dt);

__global__ 
void integrate_func3D(Qcomp* spec_old, Qcomp* spec_curr, Qcomp* spec_new, 
                        Qcomp* spec_nonl,Qreal* IF, Qreal* IFh, int Nxh, int Ny, int BSZ, Qreal dt);
void integrate_func3(Field* f_old, Field* f_curr, Field* f_new, Field*f_nonl, Qreal* IF, Qreal* IFh, Qreal dt);

__global__ 
void integrate_func4D(Qcomp* spec_old, Qcomp* spec_curr, Qcomp* spec_new, 
                        Qcomp* spec_nonl,Qreal* IF, Qreal* IFh, int Nxh, int Ny, int BSZ, Qreal dt);
void integrate_func4(Field* f_old, Field* f_curr, Field* f_new, Field*f_nonl, Qreal* IF, Qreal* IFh, Qreal dt);

#endif // end of RK4.h