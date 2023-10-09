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
#define BSZ 4

using namespace std;

int Nx = 16;
int Ny = 16;
int Nxh = Nx/2+1;
float Lx = 2*M_PI;
float Ly = 2*M_PI;
float dx = 2*M_PI/Nx;
float dy = 2*M_PI/Ny;
dim3 dimGrid  (int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
dim3 dimBlock (BSZ, BSZ); 
cufftHandle transf;
cufftHandle inv_transf;
	
cufft_error_func( cufftPlan2d( &transf, Ny, Nx, CUFFT_R2C ) );
cufft_error_func( cufftPlan2d( &inv_transf, Ny, Nx, CUFFT_C2R ) );

// this struct stores the information about the domain, including the diffential related
// values like wavenumbers and the value \alpha determining the length of the domain. 
typedef struct Mesh{
    int Nx; int Ny; int Nxh;
    float Lx; float Ly;

    float dx; float dy;
    float *kx; float *ky; float* k_squared;
    float alphax; float alphay;

    Mesh(int Nx, int Ny, float Lx, float Ly):Nx(Nx), Ny(Ny), Lx(Lx), 
    Ly(Ly), Nxh(Nx/2+1),dx(2*M_PI/Nx), dy(2*M_PI/Ny),alphax(2*M_PI/Lx),alphay(2*M_PI/Ly){

        cuda_error_func(cudaMallocManaged( &(this->kx), sizeof(float)*(Nx)));
        cuda_error_func(cudaMallocManaged( &(this->ky), sizeof(float)*(Ny)));
        cuda_error_func(cudaMallocManaged( &(this->k_squared), sizeof(float)*(Ny*Nxh)));
        for (int i=0; i<Nx; i++)          
        {
            if (i<=Nx/2+1)
            this->kx[i] = i*2*M_PI/this->alphax;
            else 
            this->kx[i] = (i-Nx)*2*M_PI/this->alphax;
        } 
        for (int j=0; j<Ny; j++)          
        {
            if(j<+Nx/2+1)
            this->ky[j] = j*2*M_PI/this->alphay;
            else 
            this->ky[j] = (j-Ny)*2*M_PI/this->alphay;
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


__global__ void D_init(float *t, int Nx, int Ny, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i < Nx && j < Ny){
        t[index] = 5*sin((float)j*dy) + 3*cos((float)j*dy) - 2*sin((float)i*dx);
    }
}
inline void init(float *t){
    D_init<<<dimGrid, dimBlock>>>(t,Nx,Ny,dx,dy);
}

__global__ void xDeriv(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;

    if(i<Nxh && j<Ny){
        if(i==0 && j==0){
            dft[index].x = 0;
            dft[index].y = 0;
        }
        else{
            dft[index] = ft[index]*im()*kx[i];
        }
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
        if(i==0 && j==0){
            dft[index].x = 0;
            dft[index].y = 0;
        }
        else{
            dft[index] = t[index]*im()*ky[j];
        }
    }
}
inline void yDeriv(cuComplex *ft, cuComplex *dft, Mesh &mesh){
    yDeriv<<<dimGrid, dimBlock>>>(ft,dft,mesh.ky,mesh.Nxh, mesh.Ny);
}


__global__ void laplacian_func(cuComplex *ft, cuComplex *lft, float* k_squared){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        lft[index] = (-1)*k_squared[index]*ft[index];
    }
}
inline void laplacian_func(cuComplex *ft, cuComplex *lft, Mesh &mesh){
    laplacian_func<<<dimGrid, dimBlock>>>(ft,lft,mesh.k_squared);
}

__global__ void poisson_solver(field &F, field &r, Mesh &mesh){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        if (i==0 && j==0){
            r.spec[index] = make_cuComplex(0. ,0.); 
        }
        else{
            r.spec[index] = (-1)*F.spec[index]/mesh.k_squared[index]; 
        }
    }
}
inline void poisson_solver(field &F, field &r, Mesh &mesh){
    poisson_solver<<<dimGrid, dimBlock>>>(F,r,mesh);
}

__global__ void F_init(field &F, Mesh &mesh){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    if(i<Nx && j<Ny){
        float x = i*mesh.dx;
        float y = j*mesh.dy;
        F.phys[index] = exp( -10*(x*x+y*y) );
    }
}
inline void F_init(field &F, Mesh &mesh){
    F_init<<<dimGrid, dimBlock>>>(F,mesh);
    BwdTrans(inv_transf, )
}
__global__ void coeff(float *f, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]/(Nx*Ny);
    }
}
inline void FwdTrans(cufftHandle transf, field &f){
    cufft_error_func( cufftExecR2C(transf, f.phys, f.spec));
}
inline void BwdTrans(cufftHandle inv_transf, field &f){
    cufft_error_func( cufftExecC2R(inv_transf, f.spec, f.phys));
    int Nx = f.mesh->Nx;
    int Ny = f.mesh->Ny;
    coeff<<<dimGrid, dimBlock>>>(f.phys, Nx, Ny);
}


void visual(field &f){
    Mesh* mesh = f.mesh;
    ofstream xcoord;
    ofstream ycoord;
    ofstream pval;
    xcoord.open("x.csv");
    ycoord.open("y.csv");
    pval.open("p.csv");

    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nx; i++){
            xcoord << i*mesh->dx << ",";
            ycoord << j*mesh->dy << ",";
            pval << f.phys[j*Nx+i] << ",";
        }
        xcoord << endl;
        ycoord << endl;
        pval << endl;
    }
    xcoord.close();
    ycoord.close();
    pval.close();
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
    Mesh mesh();
    return 0;
}