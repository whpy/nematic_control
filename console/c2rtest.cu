#include <iostream>
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
int Nx = 8;
int Ny = 8;
int Nxh = Nx/2+1;
float Lx = 2*M_PI;
float Ly = 2*M_PI;
float dx = 2*M_PI/Nx;
float dy = 2*M_PI/Ny;
dim3 dimGrid  (int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
dim3 dimBlock (BSZ, BSZ); 

typedef struct session{
    int Nx;
    int Ny;
    float Lx;
    float Ly;
    
    float dx;
    float dy;

    float *kx;
    float *ky;
    
    float alphax;
    float alphay;

    session(int Nx, int Ny, float Lx, float Ly):Nx(Nx), Ny(Ny), Lx(Lx), 
    Ly(Ly),dx(2*M_PI/Nx), dy(2*M_PI/Ny),alphax(2*M_PI/Lx),alphay(2*M_PI/Ly){

        cuda_error_func(cudaMallocManaged( &(this->kx), sizeof(float)*(Nx)));
        cuda_error_func(cudaMallocManaged( &(this->ky), sizeof(float)*(Ny)));
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
    }
} session;

typedef struct field{
    session* mesh;
    float* phys;
    cuComplex* spec;
    field(session* Mesh):mesh(Mesh){
        cuda_error_func(cudaMallocManaged(&(this->phys), sizeof(float)*((mesh->Nx)*(mesh->Ny))));
        cuda_error_func(cudaMallocManaged(&(this->spec), sizeof(cuComplex)*((mesh->Ny)*(mesh->Nx/2+1))));
    }
}field;

__global__ void D_exact(float *t, int Nx, int Ny, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i < Nx && j < Ny){
        t[index] = cos((float)j*dy);
    }
}

// void exact(float *t, session *s){
//     D_exact<<<dimGrid, dimBlock>>>(t, s->Nx, s->Ny, s->dx, s->dy);
// }

__global__ void D_init(float *t, int Nx, int Ny, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i < Nx && j < Ny){
        t[index] = sin((float)j*dy);
    }
}

inline void init(float *t){
    D_init<<<dimGrid, dimBlock>>>(t,Nx,Ny,dx,dy);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}


void print_float(float* t, int Nx, int Ny) {
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            cout <<t[j*Nx+i] << ",";
        }
        cout << endl;
    }
}

void exact(float *t, int Nx, int Ny, float dx,float dy){
    D_exact<<<dimGrid, dimBlock>>>(t, Nx, Ny, dx, dy);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__ void Dx(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny){
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

__global__ void Dy(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        if(i==0 && j==0){
            dft[index].x = 0;
            dft[index].y = 0;
        }
        else{
            dft[index] = dft[index] = ft[index]*im()*ky[j];
        }
    }
}
__global__ void coeff(float *f, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]/(Nx*Ny);
    }
}
inline void FwdTrans(cufftHandle transf, float* f, cuComplex* ft, int Nx, int Ny){
    cufft_error_func( cufftExecR2C(transf, f, ft));
}
inline void BwdTrans(cufftHandle inv_transf, cuComplex* ft, float* f, int Nx, int Ny){
    cufft_error_func( cufftExecC2R(inv_transf, ft, f));
    coeff<<<dimGrid, dimBlock>>>(f, Nx, Ny);
}
int main(){
    float *test;
    cuComplex* htest;
    cuComplex* tmp;
    float *kx;
    float *ky;
    float alpha = 2* M_PI;
    float* exa;
    cuda_error_func(cudaMallocManaged( &test, sizeof(float)*(Nx*Ny) ) );
    cuda_error_func(cudaMallocManaged( &htest, sizeof(cuComplex)*(Nxh*Ny) ) );
    cuda_error_func(cudaMallocManaged( &tmp, sizeof(cuComplex)*(Nxh*Ny) ) );
    cuda_error_func(cudaMallocManaged( &exa, sizeof(float)*(Nx*Ny) ) );
    cuda_error_func(cudaMallocManaged( &kx, sizeof(cuComplex)*(Nxh) ) );
    cuda_error_func(cudaMallocManaged( &ky, sizeof(cuComplex)*(Ny) ) );
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    for(int i = 0; i < Nxh; i++)
    {
        kx[i] = 1 * i;
    }
    
    for (int i=0; i<=Ny/2; i++)          
	{
	   ky[i] = i*2*M_PI/alpha;
    } 
	for (int i=Ny/2+1; i<Ny; i++)          
	{
        ky[i] = (i - Ny) * 1;
	}

    init(test);
    cout << "input: " << endl;
    print_float(test,Nx,Ny);
    exact(exa,Nx,Ny,dx,dy);
    cout << endl;
    cout << "exact: " << endl;
    
    print_float(exa,Nx,Ny);

    cufftHandle transf;
	cufftHandle inv_transf;
	
	cufft_error_func( cufftPlan2d( &transf, Ny, Nx, CUFFT_R2C ) );
	cufft_error_func( cufftPlan2d( &inv_transf, Ny, Nx, CUFFT_C2R ) );

    // cufft_error_func( cufftExecR2C(transf, test, htest) );
    // cufft_error_func( cufftExecC2R(inv_transf, htest, test) );
    FwdTrans(transf, test, htest, Nx, Ny);
    BwdTrans(inv_transf, htest, test, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    cout<< endl;
    cout << "after forward and inverse: " << endl;
    print_float(test,Nx,Ny);

    Dx<<<dimGrid,dimBlock>>>(htest, tmp, kx, Nxh, Ny);
    // cuda_error_func( cudaPeekAtLastError() );
	// cuda_error_func( cudaDeviceSynchronize() );

    // cufft_error_func( cufftExecC2R(inv_transf, tmp, test) );
    BwdTrans(inv_transf, tmp, test, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    cout<< endl;
    cout << "Dx: " << endl;
    print_float(test,Nx,Ny);

    Dy<<<dimGrid,dimBlock>>>(htest, tmp, ky, Nxh, Ny);
    // cuda_error_func( cudaPeekAtLastError() );
	// cuda_error_func( cudaDeviceSynchronize() );
    BwdTrans(inv_transf, tmp, test, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    cout<< endl;
    cout << "Dy: " << endl;
    print_float(test,Nx,Ny);

    // cufft_error_func( cufftExecR2C(transf, test, htest) );
    // cufft_error_func( cufftExecC2R(inv_transf, htest, test) );
    FwdTrans(transf, test, htest, Nx, Ny);
    BwdTrans(inv_transf, htest, test, Nx, Ny);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
    cout<< endl;
    cout << "after forward and inverse: " << endl;
    print_float(test,Nx,Ny);
    return 0;
}