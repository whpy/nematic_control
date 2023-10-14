#include <iostream>
#include <fstream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <device_launch_parameters.h>
#include <cufft.h>
#include <cuComplex.h>
#include "cuComplexBinOp.h"
#include "cudaErr.h"
#include "FldOp.h"

#define M_PI 3.14159265358979323846
// amount of kernels called, block_size must be n*32
#define BSZ 16
// updata tested
using namespace std;

int Nx = 512;
int Ny = 512;
int Nxh = Nx/2+1;
float Lx = 2*M_PI;
float Ly = 2*M_PI;
float dx = 2*M_PI/Nx;
float dy = 2*M_PI/Ny;
dim3 dimGrid  (int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
dim3 dimBlock (BSZ, BSZ); 
	


// this struct stores the information about the domain, including the diffential related
// values like wavenumbers and the value \alpha determining the length of the domain. 
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

        dimGridp = dim3(int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
        dimBlockp = dim3(BSZ, BSZ);

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
// operators for field solvers
__global__ void coeff(float *f, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]/(Nx*Ny);
    }
}

//update the spectral space based on the value in physical space
inline void FwdTrans(cufftHandle transf, field &f){
    cufft_error_func( cufftExecR2C(transf, f.phys, f.spec));
}
//update the physics space based on the value in spectral space
inline void BwdTrans(cufftHandle inv_transf, field &f){
    int Nx = f.mesh->Nx;
    int Ny = f.mesh->Ny;
    cufft_error_func( cufftExecC2R(inv_transf, f.spec, f.phys));
    coeff<<<f.mesh->dimGridp, f.mesh->dimBlockp>>>(f.phys, Nx, Ny);
}

__global__ void xDeriv(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*kx[i];
    }
}
inline void xDeriv(cuComplex *ft, cuComplex *dft, Mesh &mesh){
    xDeriv<<<dimGrid, dimBlock>>>(ft,dft,mesh.kx, mesh.Nxh, mesh.Ny);
}

__global__ void yDeriv(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*ky[j];
    }
}
inline void yDeriv(cuComplex *ft, cuComplex *dft, Mesh &mesh){
    yDeriv<<<dimGrid, dimBlock>>>(ft,dft,mesh.ky,mesh.Nxh, mesh.Ny);
}

__global__ void laplacian_func(cuComplex *ft, cuComplex *lft, int Nxh, int Ny, float* k_squared){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        lft[index] = (-1)*k_squared[index]*ft[index];
    }
}
inline void laplacian_func(cuComplex *ft, cuComplex *lft, Mesh &mesh){
    laplacian_func<<<dimGrid, dimBlock>>>(ft,lft,mesh.Nxh, mesh.Ny, mesh.k_squared);
}

__global__ void constant_cancel(float* f, float constant, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]-constant;
    }
}
__global__ void possion_solver_D(cuComplex* F_spec, cuComplex* r_spec, int Nxh, int Ny, float* k_squared){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        float k2 = k_squared[index];
        if (i==0 && j==0){k2 = 1.0f;}// r_spec[index] = make_cuComplex(0.,0.);
        r_spec[index] = -1.*(F_spec[index])/k2; 
    }
}

inline void possion_solver(field &F, field &r, Mesh &mesh){
    possion_solver_D<<<dimGrid, dimBlock>>>(F.spec, r.spec, mesh.Nxh, mesh.Ny, mesh.k_squared);
    BwdTrans(mesh.inv_transf, r);
}

__global__ void F_initD(float* F_phys, int Nx, int Ny, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    float s = 0.1;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        float r2 = ((x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI));
        F_phys[index] = (r2-2*s)/(s*s)*exp(-r2/(2*s));
    }
}
inline void F_init(field &F, Mesh &mesh){
    F_initD<<<dimGrid, dimBlock>>>(F.phys, mesh.Nx, mesh.Ny, mesh.dx, mesh.dy);
    // for (int j=0; j<mesh.Ny; j++){
    //     for (int i=0; i<mesh.Nx; i++){
    //         int index = i + j*mesh.Nx;
    //         float x = i*mesh.dx;
    //         float y = j*mesh.dy;
    //         float r2 = ((x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI));
    //         F.phys[index] = 4*(r2-1)*exp(-10*r2);
    //     }
    // }
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    FwdTrans(mesh.transf, F);
    
}

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
    
    Mesh s(Nx, Ny, Lx, Ly);
    coord(s);
    cout << "Lx: " << s.Lx << endl <<
    "Ly: " << s.Ly << endl <<
    "Nx: " << s.Nx << endl <<
    "Ny: " << s.Ny << endl <<
    "dx: " << s.dx << endl <<
    "dy: " << s.dy << endl <<
    "alphax: " << s.alphax << endl <<
    "alphay: " << s.alphay << endl;

    field F(&s);
    field r(&s);
    //F_initD<<<dimGrid, dimBlock>>>(F,s);
    
    //FwdTrans(s.transf, F);
    F_init(F,s);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    field_visual(F,"F1.csv");
    FwdTrans(s.transf, F);
    BwdTrans(s.inv_transf,F);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    field_visual(F,"F2.csv");
    possion_solver(F,r,s);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    float constant = r.phys[0];
    constant_cancel<<<dimGrid, dimBlock>>>(r.phys, constant, s.Nx, s.Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    // float constant = r.phys[0];
    // for(int i=0;i<s.Nx;i++) {
    //     for(int j=0;j<s.Ny;j++) {
    //         int index = i + j*s.Nx;
    //         r.phys[index] = r.phys[index]-constant;
    //     }
    // }
    field_visual(r,"r.csv");
    cout << "("<< r.spec[0].x << "," << r.spec[0].y<< ")"<< endl;
    return 0;
}
