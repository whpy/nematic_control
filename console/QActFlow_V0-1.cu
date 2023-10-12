#include <iostream>
#include <fstream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <cufft.h>
#include <cuComplex.h>
#include "cuComplexBinOp.h"
#include "cudaErr.h"

#define M_PI 3.14159265358979323846
// amount of kernels called, block_size must be n*32
#define BSZ 16

using namespace std;
int Ns = 10000;
int Nx = 512; // same as colin
int Ny = 512;
int Nxh = Nx/2+1;
float Lx = 2*M_PI;
float Ly = 2*M_PI;
float dx = 2*M_PI/Nx;
float dy = 2*M_PI/Ny;
float dt = 0.00005; // same as colin
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

        for (int j=0; j<Ny; j++){
            for (int i=0; i<Nxh; i++){
                int c = i + j*Nxh;
                this->k_squared[c] = kx[i]*kx[i] + ky[j]*ky[j];
            }
        }
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
// operators for field solvers
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
    coeff<<<dimGrid, dimBlock>>>(f.phys, Nx, Ny);
    // in the referenced source code, they seem a little bit abuse synchronization, this
    // may be a point that we could boost the performance in the future. we temporarily
    // comply with the same principle that our code would at least perform no worse than theirs
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}


__global__
// phys field multiplication: pc(x) = C*pa(x,y)*pb(x,y). prefix p denotes physical
void FldMul(float* pa, float* pb, float C, float* pc, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = C*pa[index]*pb[index];
    }
}

__global__ 
// phys field addition: pc(x,y) = a*pa(x,y) + b*pb(x,y). prefix p denotes physical
void FldAdd(float* pa, float* pb, float a, float b, float* pc, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b*pb[index];
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

__global__ void xDerivD(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*kx[i];
    }
}
inline void xDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh){
    xDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh.kx, mesh->Nxh, mesh->Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__ void yDerivD(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*ky[j];
    }__global__
}
inline void yDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh){
    yDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->ky,mesh->Nxh, mesh->Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

// maintain the symmetry of only k = 0 wavenumber
__global__ void reality_func(cuComplex *ft);
//calculate the frequently used laplacian term in non-linear function
__global__ void laplacian_funcD(cuComplex *ft, cuComplex *lft, int Nxh, int Ny, float* k_squared){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        lft[index] = (-1)*k_squared[index]*ft[index];
    }
}
inline void laplacian_funcD(cuComplex *ft, cuComplex *lft, Mesh &mesh){
    laplacian_funcD<<<dimGrid, dimBlock>>>(ft,lft,mesh.Nxh, mesh.Ny, mesh.k_squared);
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
        u_spec[index] = -1.f * ky[j]*w_spec[index]/k_squared[index];
        v_spec[index] = kx[i]*w_spec[index]/k_squared[index];
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
void S_func(float* r1, float*r2, float* S){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        S[index] = 2*sqrt(r1[index]*r1[index] + r2[index]*r2[index]);
    }
}
// non-linear terms


//necessary preparation for calculating the non-linear terms
// where nonl_r1 = nonl_r1(w,u,v,r_2,r_1)
void nonl_func0(field r1, field r2, field w, field u, field v, field S){
    // obtain the physical values of velocities and r_i
    vel_func(w, u, v);
    BwdTrans(r1.mesh,r1.spec, r1.phys);
    BwdTrans(r2.mesh,r2.spec, r2.phys);
    // calculate the physical val of S
    S_func<<<dimGrid, dimBlock>>>(r1.phys, r2.phys, S.phys);
}
//calculate the spectral of first term (\lambda S\frac{\partial u}{\partial x})of the non-linear r1
__global__
void r1nonl_funcD(float* r1nonl, float* r1, float* r2, float* w, float* u, float* v){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        r1nonl[index]
    }
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
    FldMul(r1nonl_appr.phys, u.phys, -1.f, r1nonl_appr.phys, Nx, Ny);
    // r1nonl.phys =
    // \lambda S\frac{\partial u}{\partial x}  + (-\omega_z* r2) + (-1*cn^2/Pe*S*S*r1) 
    // + (-1*D_x(w)*u(x,y))
    FldAdd(r1nonl.phys, r1nonl_appr.phys, 1.0, 1.0, r1nonl.phys, Nx, Ny);

    //(-1*v*D_y(\omega_z))
    // r1nonl_appr.spec = i*ky*w
    yDeriv(w.spec, r1nonl_appr.spec, w.mesh);




}
__global__ void r2nonl_funcD(cuComplex *, cuComplex *);
__global__ void wnonl_funcD(cuComplex *, cuComplex *);

//RK4 integrating steps
__global__ void integrate_func1(cuComplex *, cuComplex *, cuComplex);
__global__ void integrate_func2(cuComplex *, cuComplex *, cuComplex);
__global__ void integrate_func3(cuComplex *, cuComplex *, cuComplex);
__global__ void integrate_func4(cuComplex *, cuComplex *, cuComplex);

//precomputation
__global__
void precompute_func(field r1, field r2, field w, field alpha, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        r1.phys[index] = ;
        r2.phys[index] = ;
        w.phys[index] = ;
        alpha.phys[index] = ;
    }
}

__global__
void r1lin_func(float* IFr1h, float* IFr1, float* k_squared, float Pe, float cn, float dt)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        IFr1h[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt/2);
        IFr1[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt);
    }
}
__global__
void r2lin_func(float* IFr2h, float* IFr2, float* k_squared, float Pe, float cn, float dt)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        IFr2h[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt/2);
        IFr2[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt);
    }
}
__global__
void wlin_func(float* IFr1h, float* IFr1, float* k_squared, float Re, float cf, float dt)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        IFr1h[index] = exp((1.f/Re*(-1.f*(k_squared[index])-cf*cf))*dt/2);
        IFr1[index] = exp((1.f/Re*(-1.f*(k_squared[index])-cf*cf))*dt);
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

int main(){
    Mesh mesh(Nx, Ny, Lx, Ly);
    // Q_11, consistert with symbol in colin's
    field r1(&mesh), r1new(&mesh), r1appr(&mesh);
    // Q_12 (Q_21)
    field r2(&mesh), r2new(&mesh), r2appr(&mesh);
    // vorticity field \omega_z
    field w(&mesh), wnew(&mesh), wappr(&mesh);
    // activity field \alpha, this may be a 
    //variable field in the future
    field alpha(&mesh); 

    // auxiliary fields
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



    // precompute the initial conditions of three variables and parameters initialization
    precompute_func(r1, r2, w, alpha, mesh.Nx, mesh.Ny);
    // the linear integrate factors of each field, including whole time step and half time step
    float* IFr1, IFr1h, IFr2, IFr2h, IFw, IFwh;

    for (int m = 0; m <Ns; m++) {

    }




    return 0;
}