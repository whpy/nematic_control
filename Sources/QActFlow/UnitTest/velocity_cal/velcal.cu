#include <iostream>
#include <fstream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

//#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include "cuComplexBinOp.h"
#include "cudaErr.h"
#include "FldOp.h"

#define M_PI 3.14159265358979323846
// amount of kernels called, block_size must be n*32
#define BSZ 16

using namespace std;
int Ns = 1000;
int Nx = 512; // same as colin
int Ny = 512;
int Nxh = Nx/2+1;
float Lx = 2*M_PI;
float Ly = 2*M_PI;
float dx = 2*M_PI/Nx;
float dy = 2*M_PI/Ny;
float dt = 0.005; // same as colin
dim3 dimGrid  (int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
dim3 dimBlock (BSZ, BSZ); 


/******************************************************************************* 
 * data structures                                                             * 
 *******************************************************************************/

// this struct stores the information about the domain, including the diffential related
// values like wavenumbers and the value \alpha determining the length of the domain. 
typedef struct params{
    // the common non-dimensional values
    float Re, Pe, Er;
    // other non-dimensional values
    float lambda, cf, cn;
} params;

typedef struct Mesh{
    int Nx; int Ny; int Nxh;
    float Lx; float Ly;

    float dx; float dy;
    float *kx; float *ky; float* k_squared;
    // \alpha_{i} = \frac{2\pi}{L_{i}}, which determines the wavenumber factor while deriving
    float alphax; float alphay; 

    cufftHandle transf;
    cufftHandle inv_transf;

    // thread information for physical space 
    dim3 dimGridp;
    dim3 dimBlockp;
    // thread information for spectral space
    dim3 dimGridsp;
    dim3 dimBlocksp;

    Mesh(int pNx, int pNy, float pLx, float pLy):Nx(pNx), Ny(pNy), Lx(pLx), 
    Ly(pLy), Nxh(pNx/2+1),dx(2*M_PI/pNx), dy(2*M_PI/pNy),alphax(2*M_PI/pLx),alphay(2*M_PI/pLy){
        cufft_error_func( cufftPlan2d( &(this->transf), Ny, Nx, CUFFT_R2C ) );
        cufft_error_func( cufftPlan2d( &(this->inv_transf), Ny, Nx, CUFFT_C2R ) );

        cuda_error_func(cudaMallocManaged( &(this->kx), sizeof(float)*(Nx)));
        cuda_error_func(cudaMallocManaged( &(this->ky), sizeof(float)*(Ny)));
        cuda_error_func(cudaMallocManaged( &(this->k_squared), sizeof(float)*(Ny*Nxh)));
        for (int i=0; i<Nxh; i++)          
        {
            this->kx[i] = i*alphax;
        } 
        for (int j=0; j<Ny; j++)          
        {
            if(j<+Nx/2+1)
            this->ky[j] = j*alphay;
            else 
            this->ky[j] = (j-Ny)*alphay;
        } 
        // the k^2 = kx^2+ky^2
        for (int j=0; j<Ny; j++){
            for (int i=0; i<Nxh; i++){
                int c = i + j*Nxh;
                this->k_squared[c] = kx[i]*kx[i] + ky[j]*ky[j];
            }
        }

        // thread information for physical space
        dimGridp = dim3(int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
        dimBlockp = dim3(BSZ, BSZ);
        // thread information for spectral space
        dimGridsp = dim3(int((Nxh-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
        dimBlocksp = dim3(BSZ, BSZ);
    }
} Mesh;

// the template of field variables to be solved or constant 
typedef struct field{
    Mesh* mesh;
    float* phys;
    cuComplex* spec;
    field(Mesh* pMesh):mesh(pMesh){
        cuda_error_func(cudaMallocManaged(&(this->phys), sizeof(float)*((mesh->Nx)*(mesh->Ny))));
        cuda_error_func(cudaMallocManaged(&(this->spec), sizeof(cuComplex)*((mesh->Ny)*(mesh->Nxh))));
    }
}field;

void unallocate(field *f){
    cudaFree(f->phys);
    cudaFree(f->spec);
    delete f;
}
/******************************************************************************* 
 * function predefine                                                          * 
 *******************************************************************************/

// operators for field solvers
// divide a factor after transforming from spectral to physical
__global__ void coeff(float *f, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]/(Nx*Ny);
    }
}

/******************************************************************************* 
 * operator functions                                                          * 
 *******************************************************************************/

//update the spectral space based on the value in physical space
inline void FwdTrans(Mesh* pmesh, float* phys, cuComplex* spec){
    cufft_error_func( cufftExecR2C(pmesh->transf, phys, spec));
    cuda_error_func( cudaDeviceSynchronize() );
}

//update the physics space based on the value in spectral space
inline void BwdTrans(Mesh* pmesh, cuComplex* spec, float* phys){
    int Nx = pmesh->Nx;
    int Ny = pmesh->Ny;
    cufft_error_func( cufftExecC2R(pmesh->inv_transf, spec, phys));
    cuda_error_func( cudaDeviceSynchronize() );
    coeff<<<dimGrid, dimBlock>>>(phys, Nx, Ny);
    // in the referenced source code, they seem a little bit abuse synchronization, this
    // may be a point that we could boost the performance in the future. we temporarily
    // comply with the same principle that our code would at least perform no worse than theirs
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__
// __Device: phys field multiplication: pc(x) = C*pa(x,y)*pb(x,y). prefix p denotes physical
void FldMul(float* pa, float* pb, float C, float* pc, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = C*pa[index]*pb[index];
    }
}
__global__ 
// __Device: phys field addition: pc(x,y) = a*pa(x,y) + b*pb(x,y). prefix p denotes physical
void FldAdd(float* pa, float* pb, float a, float b, float* pc, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b*pb[index];
    }
}
__global__
void FldAdd(float a, float* pa, float b, float* pb, float* pc, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b*pb[index];
    }
}
__global__
// set physical field equals to a constant field
void FldSet(float * pf, float c, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if (i<Nx && j<Ny){
        pf[index] = c;
    }
}
__global__
// set two physical field equals
void FldSet(float * pf, float* c, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if (i<Nx && j<Ny){
        pf[index] = c[index];
    }
}
__global__
void SpecSet(cuComplex * pa, cuComplex* pb, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        pa[index] = pb[index];
    }
}

__global__
void SpecSet(cuComplex * pa, cuComplex c, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        pa[index] = c;
    }
}

__global__ 
// spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
void SpecAdd(cuComplex* spa, cuComplex* spb, float a, float b, cuComplex* spc, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = a* spa[index] + b* spb[index];
    }
}
__global__
void SpecAdd(float a, cuComplex* spa, float b, cuComplex* spb, cuComplex* spc, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = a* spa[index] + b* spb[index];
    }
}
__global__ void xDerivD(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*kx[i];
    }
}
inline void xDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh){
    xDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->kx, mesh->Nxh, mesh->Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__ void yDerivD(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*ky[j];
    }
}
inline void yDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh){
    yDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->ky,mesh->Nxh, mesh->Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

// maintain the symmetry of only k = 0 wavenumber
__global__ void reality_func(cuComplex *spec, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    cuComplex mean_value{ 0.f, 0.f };
    if(j<Ny && i == 0){
        mean_value = 0.5f * ( spec[index] + cuConjf(spec[Nxh*Ny-index]) );
        spec[index] = mean_value;
		spec[Nxh*Ny-index] = cuConjf(mean_value);
    }
}
//calculate the frequently used laplacian term in non-linear function(only acts on spectral)
__global__ void laplacian_funcD(cuComplex *ft, cuComplex *lft, int Nxh, int Ny, float* k_squared){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        lft[index] = (-1)*k_squared[index]*ft[index];
    }
}
inline void laplacian_func(cuComplex *ft, cuComplex *lft, Mesh* mesh){
    laplacian_funcD<<<dimGrid, dimBlock>>>(ft,lft,mesh->Nxh, mesh->Ny, mesh->k_squared);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__ void vel_funcD(cuComplex* w_spec, cuComplex* u_spec, cuComplex* v_spec, 
                            float* k_squared, float* kx, float*ky, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i==0 && j==0)
    {
        u_spec[index] = make_cuComplex(0.,0.);
        v_spec[index] = make_cuComplex(0.,0.);
    }
    else if(i<Nxh && j<Ny){
        u_spec[index] = -1.f * ky[j]*im()*w_spec[index]/(-1.f*k_squared[index]);
        v_spec[index] = kx[i]*im()*w_spec[index]/(-1.f*k_squared[index]);
    }
}
inline void vel_func(field w, field u, field v){
    int Nxh = w.mesh->Nxh;
    int Ny = w.mesh->Ny;
    float* k_squared = w.mesh->k_squared;
    float* kx = w.mesh->kx;
    float* ky = w.mesh->ky;
    vel_funcD<<<dimGrid, dimBlock>>>(w.spec, u.spec, v.spec, k_squared, kx, ky, Nxh, Ny);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );
    BwdTrans(u.mesh, u.spec, u.phys);
    BwdTrans(v.mesh, v.spec, v.phys);
}

// calculate the S field 
__global__ 
void S_func(float* r1, float*r2, float* S, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        S[index] = 2*sqrt(r1[index]*r1[index] + r2[index]*r2[index]);
    }
}
// non-linear terms


// necessary preparation for calculating the non-linear terms of three variables: w, r1, r2
// the integrate_func s would only update the spectral, so we need to calculate the physical
// value of the intermediate value like a_n, (u_{n}+a_{n}/2)*exp(alpha*dt/2), (u_{n}*exp(alpha*dt/2) + b_{n}/2) .etc
// in addition, this function could alse be used to evaluate the variables we concerned
void curr_func(field r1curr, field r2curr, field wcurr, field u, field v, field S){
    // obtain the physical values of velocities and r_i
    int Nx = r1curr.mesh->Nx;
    int Ny = r1curr.mesh->Ny;
    vel_func(wcurr, u, v);
    BwdTrans(r1curr.mesh,r1curr.spec, r1curr.phys);
    BwdTrans(r2curr.mesh,r2curr.spec, r2curr.phys);
    // calculate the physical val of S
    S_func<<<dimGrid, dimBlock>>>(r1curr.phys, r2curr.phys, S.phys, Nx, Ny);
}

inline void r1nonl_func(field r1nonl, field r1nonl_appr, field r1, field r2, field w, 
                        field u, field v, field S, float lambda, float cn, float Pe){
    // non-linear for r1: 
    // \lambda S\frac{\partial u}{\partial x}  + (-1* \omega_z* r2) + (-cn^2/Pe *S^2*r1)
    // + (-1* u* D_x\omega_z) + (-1*v*D_y(\omega_z))
    int Nx = r1.mesh->Nx;
    int Ny = r1.mesh->Ny;

    // \lambda S\frac{\partial u}{\partial x}
    //nonl1_appr.spec = \partial_x u
    xDeriv(u.spec, r1nonl_appr.spec, r1nonl.mesh);
    //nonl1_appr.phys = \partial_x u
    BwdTrans(r1nonl.mesh, r1nonl_appr.spec, r1.phys);
    // r1nonl.phys = \lambda*S(x,y) * r1nonl_appr = \lambda*S(x,y) * \partial_x u(x,y) 
    FldMul<<<dimGrid, dimBlock>>>(r1nonl_appr.phys, S.phys, lambda, r1nonl.phys, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    // (-\omega_z* r2)
    // r1nonl_appr.phys = -1*\omega*r2
    FldMul<<<dimGrid, dimBlock>>>(w.phys, r2.phys, -1.0, r1nonl_appr.phys, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );
    // r1nonl.phys = \lambda*S(x,y) * \partial_x u(x,y) + (-\omega_z* r2)
    FldAdd<<<dimGrid, dimBlock>>>(r1nonl.phys, r1nonl_appr.phys, 1., 1., r1nonl.phys, Nx, Ny);

    //(-cn^2/Pe *S^2*r1)
    // r1nonl_appr.phys = -1*cn^2/Pe*S*S
    FldMul<<<dimGrid, dimBlock>>>(S.phys, S.phys, -1.*cn*cn/Pe, r1nonl_appr.phys, Nx, Ny);
    // r1nonl_appr.phys = -1*cn^2/Pe*S*S*r1
    FldMul<<<dimGrid, dimBlock>>>(r1nonl_appr.phys, r1.phys, 1.f, r1nonl_appr.phys, Nx, Ny);
    // r1nonl.phys = \lambda S\frac{\partial u}{\partial x}  + (-\omega_z* r2) + (-1*cn^2/Pe*S*S*r1)
    FldAdd<<<dimGrid, dimBlock>>>(r1nonl.phys, r1nonl_appr.phys, 1., 1., r1nonl.phys, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    //(-u*D_x(\omega_z))
    // r1nonl_appr.spec = i*kx*w
    xDeriv(w.spec, r1nonl_appr.spec, w.mesh);
    // r1nonl_appr.phys = D_x(w)
    BwdTrans(r1nonl_appr.mesh,r1nonl_appr.spec, r1nonl_appr.phys);
    // r1nonl_appr.phys = -1*r1nonl_appr.phys*u(x,y) = -1*D_x(w)*u(x,y)
    FldMul<<<dimGrid, dimBlock>>>(r1nonl_appr.phys, u.phys, -1.f, r1nonl_appr.phys, Nx, Ny);
    // r1nonl.phys =
    // \lambda S\frac{\partial u}{\partial x}  + (-\omega_z* r2) + (-1*cn^2/Pe*S*S*r1) 
    // + (-1*D_x(w)*u(x,y))
    FldAdd<<<dimGrid, dimBlock>>>(r1nonl.phys, r1nonl_appr.phys, 1.0, 1.0, r1nonl.phys, Nx, Ny);

    //(-1*v*D_y(\omega_z))
    // r1nonl_appr.spec = i*ky*w
    yDeriv(w.spec, r1nonl_appr.spec, w.mesh);
    // r1nonl_appr.phys = D_y(w)
    BwdTrans(r1nonl_appr.mesh,r1nonl_appr.spec, r1nonl_appr.phys);
    // r1nonl_appr.phys = -1*r1nonl_appr.phys*v(x,y) = -1*D_y(w)*v(x,y)
    FldMul<<<dimGrid, dimBlock>>>(r1nonl_appr.phys, v.phys, -1.f, r1nonl_appr.phys, Nx, Ny);
    // r1nonl.phys = r1nonl.phys + r1nonl_appr.phys = 
    // \lambda S\frac{\partial u}{\partial x}  + (-\omega_z* r2) + (-1*cn^2/Pe*S*S*r1) 
    // + (-1*D_x(w)*u(x,y)) + (-1*v*D_y(\omega_z))
    FldAdd<<<dimGrid, dimBlock>>>(r1nonl.phys, r1nonl_appr.phys, 1.0, 1.0, r1nonl.phys, Nx, Ny);

    // the spectral of r1 nonlinear term is calculated here based on the physical value
    // that evaluated before.
    FwdTrans(r1nonl.mesh, r1nonl.phys, r1nonl.spec);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize());
}

inline void r2nonl_func(field r2nonl, field r2nonl_appr, field r1, field r2, field w, 
                        field u, field v, field S, float lambda, float cn, float Pe){
    // non-linear for r1: 
    // \lambda* S* 1/2* (D_x(v)+D_y(u)) + (\omega_z* r1) + (-cn^2/Pe *S^2*r2)
    // + (-1* u* D_x(r2))) + (-1*v*D_y(r2))
    int Nx = r2nonl.mesh->Nx;
    int Ny = r2nonl.mesh->Ny;

    // \lambda* S* 1/2* (D_x(v))
    //nonl1_appr.spec = \partial_x u
    xDeriv(v.spec, r2nonl_appr.spec, r2nonl.mesh);
    //nonl1_appr.phys = \partial_x u
    BwdTrans(r2nonl_appr.mesh, r2nonl_appr.spec, r2nonl_appr.phys);
    // r2nonl.phys = \lambda*S(x,y) * r2nonl_appr = \lambda/2 *S(x,y) * D_x(u(x,y)) 
    FldMul<<<dimGrid, dimBlock>>>(r2nonl_appr.phys, S.phys, lambda/2, r2nonl.phys, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    // \lambda* S* 1/2 *(D_y(u))
    //nonl2_appr.spec = D_y(u)
    yDeriv(u.spec, r2nonl_appr.spec, r2nonl_appr.mesh);
    //nonl2_appr.phys = D_y(u)
    BwdTrans(r2nonl_appr.mesh, r2nonl_appr.spec, r2nonl_appr.phys);
    // r2nonl_appr.phys = \lambda/2 *S(x,y) * r2nonl_appr = \lambda/2 *S(x,y) * D_x(u(x,y)) 
    FldMul<<<dimGrid, dimBlock>>>(r2nonl_appr.phys, S.phys, lambda/2, r2nonl_appr.phys, Nx, Ny);
    // r2nonl.phys = r2nonl.phys + r2nonl_appr.phys = \lambda/2 *S(x,y) *D_x(u(x,y)) + \lambda/2 *S(x,y) * D_x(u(x,y)) 
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl.phys, r2nonl_appr.phys, 1., 1., r2nonl.phys, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    // (\omega_z* r1)
    // r1nonl_appr.phys = 1.0* \omega*r2
    FldMul<<<dimGrid, dimBlock>>>(w.phys, r2.phys, 1.0, r2nonl_appr.phys, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );
    // r2nonl.phys = (\lambda/2 *S(x,y) *D_x(u(x,y))) + (\lambda/2 *S(x,y) * D_x(u(x,y))) + (\omega_z* r2)
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl.phys, r2nonl_appr.phys, 1., 1., r2nonl.phys, Nx, Ny);

    //(-cn^2/Pe *S^2*r2)
    // r2nonl_appr.phys = -1*cn^2/Pe*S*S
    FldMul<<<dimGrid, dimBlock>>>(S.phys, S.phys, -1.*cn*cn/Pe, r2nonl_appr.phys, Nx, Ny);
    // r2nonl_appr.phys = -1*cn^2/Pe*S*S*r2
    FldMul<<<dimGrid, dimBlock>>>(r2nonl_appr.phys, r2.phys, 1.f, r2nonl_appr.phys, Nx, Ny);
    // r1nonl.phys = (\lambda/2 *S(x,y) *D_x(u(x,y))) + (\lambda/2 *S(x,y) * D_x(u(x,y))) + (\omega_z* r2) + (-1*cn^2/Pe*S*S*r2)
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl.phys, r2nonl_appr.phys, 1., 1., r2nonl.phys, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    //(-u*D_x(\omega_z))
    // r2nonl_appr.spec = i*kx*w
    xDeriv(w.spec, r2nonl_appr.spec, w.mesh);
    // r2nonl_appr.phys = D_x(w)
    BwdTrans(r2nonl_appr.mesh,r2nonl_appr.spec, r2nonl_appr.phys);
    // r2nonl_appr.phys = -1*r2nonl_appr.phys*u(x,y) = -1*D_x(w)*u(x,y)
    FldMul<<<dimGrid, dimBlock>>>(r2nonl_appr.phys, u.phys, -1.f, r2nonl_appr.phys, Nx, Ny);
    // r2nonl.phys =
    // (\lambda/2 *S(x,y) *D_x(u(x,y))) + (\lambda/2 *S(x,y) * D_x(u(x,y))) + (\omega_z* r2) 
    //+ (-1*cn^2/Pe*S*S*r2) + (-1*D_x(w)*u(x,y))
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl.phys, r2nonl_appr.phys, 1.0, 1.0, r2nonl.phys, Nx, Ny);

    //(-1*v*D_y(\omega_z))
    // r2nonl_appr.spec = i*ky*w
    yDeriv(w.spec, r2nonl_appr.spec, w.mesh);
    // r2nonl_appr.phys = D_y(w)
    BwdTrans(r2nonl_appr.mesh,r2nonl_appr.spec, r2nonl_appr.phys);
    // r2nonl_appr.phys = -1*r2nonl_appr.phys*v(x,y) = -1*D_y(w)*v(x,y)
    FldMul<<<dimGrid, dimBlock>>>(r2nonl_appr.phys, v.phys, -1.f, r2nonl_appr.phys, Nx, Ny);
    // r2nonl.phys = r2nonl.phys + r2nonl_appr.phys = 
    // (\lambda/2 *S(x,y) *D_x(u(x,y))) + (\lambda/2 *S(x,y) * D_x(u(x,y))) + (\omega_z* r2) 
    // + (-1*cn^2/Pe*S*S*r2) + (-1*D_x(w)*u(x,y)) + (-1*v*D_y(\omega_z))
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl.phys, r2nonl_appr.phys, 1.0, 1.0, r2nonl.phys, Nx, Ny);

    // the spectral of r1 nonlinear term is calculated here based on the physical value
    // that evaluated before.
    FwdTrans(r2nonl.mesh, r2nonl.phys, r2nonl.spec);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize());
}


inline void wnonl_func(field wnonl, field wnonl_appr, field appr1, field p11, field p12, field p21, field r1, field r2, field w, 
                        field u, field v, field alpha, field S, float Re, float Er, float cn, float lambda){
            // wnonl = 1/ReEr * (D^2_xx(p12) - 2*D^2_xy(p11) - D^2_yy(p21))  
            //         + (-1* u*D_x(w)) + (-1* v* D_y(w)) 
    p11nonl_func(p11, wnonl_appr, appr1, r1, r2, S, alpha, lambda, cn);
    p11nonl_func(p11, wnonl_appr, appr1, r1, r2, S, alpha, lambda, cn); 
    p12nonl_func(p12, wnonl_appr, appr1, r1, r2, S, alpha, lambda, cn); 
    p21nonl_func(p21, wnonl_appr, appr1, r1, r2, S, alpha, lambda, cn); 

    // wnonl_appr.spec = D_x(p12)
    xDeriv(p12.spec, wnonl_appr.spec, p12.mesh);
    // wnonl.spec = D^2_xx(p12)
    xDeriv(wnonl_appr.spec, wnonl.spec, wnonl_appr.mesh);
    
    // wnonl_appr.spec = D_x(p11)
    xDeriv(p11.spec, wnonl_appr.spec, p11.mesh);
    // wnonl_appr.spec = D_y(wnonl_appr.spec) = D^2_xy(p11)
    yDeriv(wnonl_appr.spec, wnonl_appr.spec, wnonl_appr.mesh);
    // wnonl.spec = D^2_xx(p12) - 2*wnonl_appr.spec = D^2_xx(p12) - 2*D^2_xy(p11)
    SpecAdd<<<dimGrid, dimBlock>>>(1., wnonl.spec, -2., wnonl_appr.spec, 
    wnonl.spec, wnonl.mesh->Nxh, wnonl.mesh->Ny);

    // wnonl_appr.spec = D_y(p21)
    yDeriv(p21.spec, wnonl_appr.spec, p21.mesh);
    // wnonl.spec = D^2_yy(p12)
    yDeriv(wnonl_appr.spec, wnonl.spec, wnonl_appr.mesh);
    // wnonl.spec = D^2_xx(p12) - 2*D^2_xy(p11) - wnonl_appr.spec 
    // = D^2_xx(p12) - 2*D^2_xy(p11) - D^2_yy(p21)
    SpecAdd<<<dimGrid, dimBlock>>>(1., wnonl.spec, -1., wnonl_appr.spec, 
    wnonl.spec, wnonl.mesh->Nxh, wnonl.mesh->Ny);

    // wnonl_appr.spec = D_x(w)
    xDeriv(w.spec, wnonl_appr.spec, wnonl_appr.mesh);
    // wnonl_appr.phys = D_x(w)
    BwdTrans(wnonl_appr.mesh, wnonl_appr.spec, wnonl_appr.phys);
    // wnonl_appr.phys = (-1* u* D_x(w))
    FldMul<<<dimGrid, dimBlock>>>(wnonl_appr.phys, u.phys, -1., wnonl_appr.phys, Nx, Ny);
    // forward to the spectral: wnonl_appr.spec = Four((-1* u* D_x(w)))
    FwdTrans(wnonl_appr.mesh, wnonl_appr.phys, wnonl_appr.spec);
    // wnonl.spec = D^2_xx(p12) - 2*D^2_xy(p11) + wnonl_appr.spec 
    // = D^2_xx(p12) - 2*D^2_xy(p11) - D^2_yy(p21) + (-1* u* D_x(w))
    SpecAdd<<<dimGrid, dimBlock>>>(1., wnonl.spec, 1., wnonl_appr.spec, 
    wnonl.spec, wnonl.mesh->Nxh, wnonl.mesh->Ny);

    // wnonl_appr.spec = D_y(w)
    yDeriv(w.spec, wnonl_appr.spec, wnonl_appr.mesh);
    // wnonl_appr.phys = D_y(w)
    BwdTrans(wnonl_appr.mesh, wnonl_appr.spec, wnonl_appr.phys);
    // wnonl_appr.phys = (-1* v* D_y(w))
    FldMul<<<dimGrid, dimBlock>>>(wnonl_appr.phys, v.phys, -1., wnonl_appr.phys, Nx, Ny);
    // forward to the spectral: wnonl_appr.spec = Four((-1* v* D_y(w)))
    FwdTrans(wnonl_appr.mesh, wnonl_appr.phys, wnonl_appr.spec);
    // wnonl.spec = D^2_xx(p12) - 2*D^2_xy(p11) + wnonl_appr.spec 
    // = D^2_xx(p12) - 2*D^2_xy(p11) - D^2_yy(p21) + (-1* u* D_x(w)) + (-1* v* D_y(w))
    SpecAdd<<<dimGrid, dimBlock>>>(1., wnonl.spec, 1., wnonl_appr.spec, 
    wnonl.spec, wnonl.mesh->Nxh, wnonl.mesh->Ny);

    cuda_error_func( cudaDeviceSynchronize() );
    // here the wnonl has updated sucessfully
    
}
// calculate the cross term in pij where Cross(r1,r2) = 2*(r2*\Delta(r1) - r1*\Delta(r2))
inline void pCross_func(field p,field appr, field r1, field r2){
    // this function only works on the physical space
    int Nx = p.mesh->Nx;
    int Ny = p.mesh->Ny;
    //appr.spec = Four(/Delta(r1))
    laplacian_func(r1.spec,appr.spec,appr.mesh);
    //appr.phys = /Delta(r1)
    BwdTrans(appr.mesh, appr.spec, appr.phys);
    // p.phys = 2* r2*\Delta(r1)
    FldMul<<<dimGrid, dimBlock>>>(appr.phys,r2.phys, 2., p.phys, Nx, Ny);

    //appr.spec = Four(/Delta(r2))
    laplacian_func(r1.spec,appr.spec,appr.mesh);
    //appr.phys = /Delta(r2)
    BwdTrans(appr.mesh, appr.spec, appr.phys);
    // appr.phys = -2* r2*\Delta(r1)
    FldMul<<<dimGrid, dimBlock>>>(appr.phys,r2.phys, -2., appr.phys, Nx, Ny);
    // p.phys = appr.phys + p.phys = -2* r2*\Delta(r1) + 2* r2*\Delta(r1)
    FldAdd<<<dimGrid, dimBlock>>>(1., p.phys, 1., appr.phys, p.phys, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    // cross term physical value update successfully
}
// the rest term of the pij where 
//Single(ri) = \alpha*ri + \lambda* S*(cn^2*(S^2-1)*ri - \Delta ri) 
//           = \alpha*ri - \lambda* S* \Delta(ri) 
//            + \lambda* S*cn^2*(S^2)*ri - \lambda* S*cn^2*ri
inline void pSingle_func(field p, field appr, field r, field S, field alpha, float lambda, float cn){
    // this function only works on the physical space
    int Nx = appr.mesh->Nx;
    int Ny = appr.mesh->Ny;
    
    //appr.phys = \alpha* ri
    FldMul<<<dimGrid, dimBlock>>>(r.phys, alpha.phys, 1.f, appr.phys, Nx, Ny);
    
    // -\lambda* S* \Delta(ri)
    // p.spec = \Delta(r)
    laplacian_func(p.spec, r.spec , p.mesh);
    // p.phys = \Delta(r)
    BwdTrans(p.mesh, p.spec, p.phys);

    // p.phys = -\lambda*\Delta(r)*alpha
    FldMul<<<dimGrid, dimBlock>>>(p.phys, S.phys, -1*lambda, p.phys, Nx, Ny);
    // p.phys = p.phys + appr.phys = \alpha* ri -\lambda*\Delta(r)*alpha
    FldAdd<<<dimGrid, dimBlock>>>(1.f, appr.phys, 1.f, p.phys, p.phys, Nx, Ny);

    // \lambda*cn^2*(S^3)*ri
    // appr.phys = \lambda* cn^2 * S * S
    FldMul<<<dimGrid, dimBlock>>>(S.phys, S.phys, lambda*cn*cn, appr.phys, Nx, Ny);
    // appr.phys =appr.phys* S = \lambda* cn^2 * S * S* S
    FldMul<<<dimGrid, dimBlock>>>(appr.phys, S.phys, 1., appr.phys, Nx, Ny);
    // appr.phys =appr.phys* ri = \lambda* cn^2 * S * S* S* ri
    FldMul<<<dimGrid, dimBlock>>>(appr.phys, r.phys, 1., appr.phys, Nx, Ny);
    // p.phys = p.phys + appr.phys = \alpha* ri -\lambda*\Delta(r)*alpha + \lambda*cn^2*S^3*ri
    FldAdd<<<dimGrid, dimBlock>>>(1.f, appr.phys, 1.f, p.phys, p.phys, Nx, Ny);

    // -\lambda* S*cn^2*ri
    // appr.phys = -1*\lambda* cn^2 * S * ri
    FldMul<<<dimGrid, dimBlock>>>(S.phys, r.phys, -1*lambda*cn*cn, appr.phys, Nx, Ny);
    // p.phys = p.phys + appr.phys 
    // = \alpha* ri -\lambda*\Delta(r)*alpha + \lambda*cn^2*S^3*ri + (-1*\lambda*cn^2 *S*ri)
    FldAdd<<<dimGrid, dimBlock>>>(1.f, appr.phys, 1.f, p.phys, p.phys, Nx, Ny);
    // cross term physical value update successfully
}
// p11 = Single(r1)
inline void p11nonl_func(field p11, field appr, field appr1, field r1, field r2, field S, 
                        field alpha, float lambda, float cn){
    // our strategy is firstly update the phys then finally update the spectral
    // p11.phys = \lambda* S(cn^2*(S^2-1)*r1 - \Delta r1) + \alpha*r1
    pSingle_func(p11, appr, r1, S, alpha, lambda,cn);
    cuda_error_func( cudaDeviceSynchronize() );
    FwdTrans(p11.mesh, p11.phys, p11.spec);
    // p11 spectral update finished
}
// p12 = Cross(r1,r2) + Single(r2)
inline void p12nonl_func(field p12, field appr, field appr1, field r1, field r2, field S, 
                        field alpha, float lambda, float cn){
    int Nx = p12.mesh->Nx;
    int Ny = p12.mesh->Ny;
    // p12.phys = Cross(r1,r2) = 2*(r2*\Delta(r1) - r1*\Delta(r2))
    pCross_func(p12, appr, r1, r2);
    // appr.phys = Single(r2) = \lambda* S(cn^2*(S^2-1)*r2 - \Delta r2) + \alpha*r2
    pSingle_func(appr, appr1, r2, S, alpha, lambda,cn);
    // p12.phys = p12.phys + appr.phys = Cross(r1,r2) + Single(r2)
    FldAdd<<<dimGrid, dimBlock>>>(1., p12.phys, 1., appr.phys, p12.phys, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    FwdTrans(p12.mesh, p12.phys, p12.spec);
    cuda_error_func( cudaDeviceSynchronize() );
    // p12 spectral update finished
}
// p22 = -1*Cross(r1,r2) + Single(r2)
inline void p21nonl_func(field p21, field appr, field appr1, field r1, field r2, field S, 
                        field alpha, float lambda, float cn){
    int Nx = p21.mesh->Nx;
    int Ny = p21.mesh->Ny;
    // p21.phys = Cross(r2,r1) = 2*(r1*\Delta(r2) - r2*\Delta(r1))
    pCross_func(p21, appr, r2, r1);
    // appr.phys = Single(r2) = \lambda* S(cn^2*(S^2-1)*r2 - \Delta r2) + \alpha*r2
    pSingle_func(appr, appr1, r2, S, alpha, lambda, cn);
    // p21.phys = p21.phys + appr.phys = Cross(r2,r1) + Single(r2)
    FldAdd<<<dimGrid, dimBlock>>>(1., p21.phys, 1., appr.phys, p21.phys, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    FwdTrans(p21.mesh, p21.phys, p21.spec);
    cuda_error_func( cudaDeviceSynchronize() );
    // p21 spectral update finished
}

//RK4 integrating steps
// du/dt = \alpha*u + F(t,u)
// IFh = exp(\alpha*dt/2). IF = exp(\alpha*dt)
// u_{n+1} = u_{n}*IF + 1/6*(a*IF + 2b*IFh + 2c*IFh + d)
// a = dt*F(t_n,u_n)
// b = dt*F(t_n+dt/2, (u_n+a/2)*IFh)
// c = dt*F(t_n+dt/2, u_n*IFh + b/2)
// d = dt*F(t_n+dt, u_n*IF + c*IFh)
__global__
void integrate_func0(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new,
                    float* IF, float* IFh, int Nxh, int Ny, float dt){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        // u_{n+1} = u_{n}*exp(alpha * dt)
        spec_new[index] = spec_old[index]*IF[index];
        // u_{n}
        spec_curr[index] = spec_old[index];
    }
}
__global__ 
void integrate_func1(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
                    float* IF, float* IFh, int Nxh, int Ny, float dt){
    // spec_nonl = a_n/dt here
    // spec_curr represents the value to be input into Nonlinear function for b_n/dt next 
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        cuComplex an = spec_nonl[index]*dt;
        // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n)
        spec_new[index] = spec_new[index] + 1/6*IF[index] * an;
        // (u_{n}+a_{n}/2)*exp(alpha*dt/2)
        spec_curr[index] = (spec_old[index]+an/2) * IFh[index];
    }
}
__global__ void integrate_func2(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt){
    // spec_nonl = b_n/dt here
    // spec_curr represents the value to be input into Nonlinear function for c_n/dt next 
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        cuComplex bn = spec_nonl[index]*dt;
        // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n) + 1/6*exp(alpha*dt/2)*(b_n)
        spec_new[index] = spec_new[index] + 1/3*IFh[index] * bn;
        // (u_{n}*exp(alpha*dt/2) + b_{n}/2)
        spec_curr[index] = (spec_old[index]*IFh[index] + bn/2) ;
    }
}
__global__ void integrate_func3(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt){
    // spec_nonl = c_n/dt here
    // spec_curr represents the value to be input into Nonlinear function for d_n/dt next 
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        cuComplex cn = spec_nonl[index]*dt;
        // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n) + 1/3*exp(alpha*dt/2)*(b_n) 
        //         + 1/3*exp(alpha*dt/2)*(c_n)
        spec_new[index] = spec_new[index] + 1/3*IFh[index] * cn;
        // u_{n}*exp(alpha*dt) + c_{n} * exp(alpha*dt/2)
        spec_curr[index] = (spec_old[index]*IF[index] + cn*IFh[index]) ;
    }
}
__global__ void integrate_func4(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, float dt){
    // spec_nonl = d_n/dt here
    // spec_curr represents the value to be input into Nonlinear function for d_n/dt next 
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        cuComplex dn = spec_nonl[index]*dt;
        // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n) + 1/3*exp(alpha*dt/2)*(b_n) 
        //         + 1/3*exp(alpha*dt/2)*(c_n) + 1/6*d_n
        spec_new[index] = spec_new[index] + 1/6*dn;
    }

}

//precomputation
// __global__
// void precompute_funcD(float* r1, float* r2, float* w, float* alpha, 
// int Nx, int Ny, float dx, float dy){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nx + i;
//     float x = i*dx;
//     float y = j*dy;
//     float rr = (x-M_PI)*(x-M_PI) + (y-M_PI)*(y-M_PI);
//     if(i<Nx && j<Ny){
//         r1[index] = -1*(cos(x)+sin(y));
//         r2[index] = 0.;
//         w [index] = cos(x)+sin(y);
//         alpha[index] = 3.5;
//     }
// }
__global__
void precompute_funcD(float* r1, float* r2, float* w, float* alpha, 
int Nx, int Ny, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    float x = i*dx;
    float y = j*dy;
    float rr = (x-M_PI)*(x-M_PI) + (y-M_PI)*(y-M_PI);
    float a = -10;
    if(i<Nx && j<Ny){
        //vel u = -1*Dy(\phi) = -cos(x)*cos(y)
        r1[index] = -1.*cos(x)*cos(y);
        //vel v = Dx(\phi) = -sin(x)*sin(y)
        r2[index] = -1.*sin(x)*sin(y);
        w [index] = -2*sin(y)*cos(x);
        alpha[index] = 3.5;
    }
}

void precompute_func(field r1, field r2, field w, field alpha){
    float dx = r1.mesh->dx;
    float dy = r1.mesh->dy;
    int Nx = r2.mesh->Nx;
    int Ny = r2.mesh->Ny;
    precompute_funcD<<<dimGrid, dimBlock>>>(r1.phys, r2.phys, w.phys, 
    alpha.phys, Nx, Ny, dx, dy);
}

__global__
void r1lin_func(float* IFr1h, float* IFr1, float* k_squared, float Pe, float cn, float dt, int Nxh, int Ny)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        IFr1h[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt/2);
        IFr1[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt);
    }
}

__global__
void r2lin_func(float* IFr2h, float* IFr2, float* k_squared, float Pe, float cn, float dt, int Nxh, int Ny)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        IFr2h[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt/2);
        IFr2[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt);
    }
}

__global__
void wlin_func(float* IFwh, float* IFw, float* k_squared, float Re, float cf, float dt, int Nxh, int Ny)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        IFwh[index] = exp(1.0*dt/2);
        IFw[index] = exp(1.0*dt);
    }
}
/******************************************************************************* 
 * visualization term                                                          * 
 *******************************************************************************/

void coord(Mesh &mesh){
    ofstream xcoord("x.csv");
    ofstream ycoord("y.csv");
    for (int j=0; j<mesh.Ny; j++){
        for ( int i=0; i< mesh.Nx; i++){
            float x = mesh.dx*i;
            float y = mesh.dy*j;
            xcoord << x << ",";
            ycoord << y << ",";
        }
        xcoord << endl;
        ycoord << endl;
    }
    xcoord.close();
    ycoord.close();
}

void field_visual(field &f, string name){
    Mesh* mesh = f.mesh;
    ofstream fval;
    string fname = name;
    fval.open(fname);
    for (int j=0; j<mesh->Ny; j++){
        for (int i=0; i<mesh->Nx; i++){
            fval << f.phys[j*Nx+i] << ",";
        }
        fval << endl;
    }
    fval.close();
}
void print_float(float* t, int Nx, int Ny) {
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            cout <<t[j*Nx+i] << ",";
        }
        cout << endl;
    }
}

void print_spec(cuComplex* t, int Nxh, int Ny) {
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nxh; i++) {
            cout <<t[j*Nxh+i].x<<","<<t[j*Nxh+i].y << "  ";
        }
        cout << endl;
    }
}
// in this unit, we test the vel_func which evaluates velocity (u,v) field by 
// vorticy field w. We here set w = -2*cos(x)*sin(y). u = cos(x)cos(y), v = -sin(x)sin(y)
int main(){
    Mesh mesh(Nx, Ny, Lx, Ly);
    // the variables to be solved, for each variable we distribute four field: 
    // storing the n step value( r1 ), storing the next step value( r1new ), 
    // storing the intermediate value for RK4( r1curr ), auxiliary field while computing( r1appr ).

    // Q_11, consistert with symbol in colin's
    field r1(&mesh), r1new(&mesh), r1curr(&mesh), r1appr(&mesh);
    // Q_12 (Q_21)
    field r2(&mesh), r2new(&mesh), r2curr(&mesh), r2appr(&mesh);
    // vorticity field \omega_z
    field w(&mesh), wnew(&mesh), wcurr(&mesh), wappr(&mesh);
    // activity field \alpha, this may be a 
    //variable field in the future
    field alpha(&mesh); 

    // the auxiliary fields
    // the velocity field u, v
    // their association with \omega_z is:
    // \hat{u} = -k_y/(|k|^2)*hat{\omega_z}, \hat{v} = k_x/(|k|^2)*hat{\omega_z)
    // where |k|^2 = k_x^2 + k_y^2
    field u(&mesh), v(&mesh);
    // the nonlinear terms
    field r1nonl(&mesh), r1nonlappr(&mesh);
    field r2nonl(&mesh), r2nonlappr(&mesh);
    field wnonl(&mesh), wnonlappr(&mesh);
    // the s field
    field S(&mesh);
    // the stress tensor fields
    field p11(&mesh), p12(&mesh), p21(&mesh);


    coord(mesh);
    // precompute the initial conditions of three variables and parameters initialization
    // precompute test
    precompute_func(r1, r2, w, alpha);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(r1, "u.csv");
    field_visual(w, "w.csv");
    field_visual(r2, "v.csv");

    FwdTrans(w.mesh, w.phys, w.spec);
    FwdTrans(r1.mesh, r1.phys, r1.spec);
    FwdTrans(r2.mesh, r2.phys, r2.spec);
    vel_func(w,u,v);
    // vel_funcD<<<dimGrid,dimBlock>>>(w.spec, u.spec, v.spec, mesh.k_squared, mesh.kx, mesh.ky, mesh.Nxh, mesh.Ny);
    // cuda_error_func( cudaDeviceSynchronize() );
    // BwdTrans(u.mesh, u.spec, u.phys);
    // BwdTrans(v.mesh, v.spec, v.phys);
    // print_spec(r1.spec,mesh.Nxh,mesh.Ny);
    // cout << "seperated" << endl;
    // print_spec(u.spec,mesh.Nxh,mesh.Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(u, "uc.csv");
    field_visual(v, "vc.csv");
    
    return 0;
}