#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <cufft.h>
#include <cuComplex.h>
#include "cuComplexBinOp.h"
#include <cudaErr.h>

#define _pi 3.14159265358979323846
// amount of kernels called, block_size must be n*32
#define BSZ 4

using namespace std;
typedef struct __session{
    int Nx;
    int Ny;
    float dx;
    float dy;
    float alphax;
    float alphay;
    float Lx;
    float Ly;
    float *kx;
    float *ky;
}session;

void session_init(session *s, int Nx, int Ny, float Lx, float Ly, float *kx, float *ky){
    s->Nx = Nx;
    s->Ny = Ny;
    s->Lx = Lx;
    s->Ly = Ly;
    s->dx = Lx/Nx;
    s->dy = Ly/Ny;
    s->alphax = 2*M_PI/Lx;
    s->alphay = 2*M_PI/Ly;
    s->kx = kx;
    s->ky = ky;
}
int Nx = 8;
int Ny = 8;
float Lx = 2*M_PI;
float Ly = 2*M_PI;
float dx = 2*M_PI/Nx;
float dy = 2*M_PI/Ny;
dim3 dimGrid  (int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
dim3 dimBlock (BSZ, BSZ); 


//cufft_error_func( cufftPlan2d( &transf, Ny, Nx, CUFFT_R2C ) );
//cufft_error_func( cufftPlan2d( &inv_transf, Ny, Nx, CUFFT_C2R ) );

__global__ void D_init(float *t, int Nx, int Ny, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i*Ny + j;
    if(i < Nx && j < Ny){
        t[index] = cos((float)j*dy);
    }
}
void init(float *t){
    D_init<<<dimGrid, dimBlock>>>(t,Nx,Ny,dx,dy);
}


void print_float(float* t, int Nx, int Ny) {
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            cout <<t[i*Ny+j] << ",";
        }
        cout << endl;
    }
}

int main(){
    float *test;
    float *kx;
    float *ky;
    cuda_error_func(cudaMallocManaged( &kx, sizeof(float)*(Nx/2+1)));
    cuda_error_func(cudaMallocManaged( &ky, sizeof(float)*Ny));
    float alpha = 2*M_PI;
    for (int i = 0; i < Nx/2; i++) {
        kx[i] = i*2*M_PI/alpha;
    }
    for (int i=Nx/2+1; i<Nx; i++){
        kx[i] = (i-Nx)*2*M_PI/alpha;
    }

    for (int j=0; j<Ny/2; j++){
        ky[j] = j*2*M_PI/alpha;
    }
    for (int j=Ny/2+1; j<Ny; j++){
        ky[j] = (j-Ny)*2*M_PI/alpha;
    }
    session s;
    session_init(&s,Nx,Ny,Lx,Ly,kx,ky);
    cout << "Lx: " << s.Lx << endl <<
    "Ly: " << s.Ly << endl <<
    "Nx: " << s.Nx << endl <<
    "Ny: " << s.Ny << endl <<
    "dx: " << s.dx << endl <<
    "dy: " << s.dy << endl <<
    "alphax: " << s.alphax << endl <<
    "alphay: " << s.alphay << endl;
    cuComplex* ctest;
    
    cufftHandle plan;
    cufftPlan2d(&plan, Nx, Ny, CUFFT_C2C);
    cuda_error_func(cudaMallocManaged( &test, sizeof(float)*Nx*Ny));
    cuda_error_func(cudaMallocManaged( &ctest, sizeof(cuComplex)*(Nx+1)/2*Ny));
    init(test);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << "input: " << endl;
    print_float(test, Nx, Ny);

    cout << endl;
    cout << "output: " << endl;
    cufftExecR2C(plan,test,ctest);
    cufftExecC2R(plan,ctest,test);
    cuda_error_func( cudaDeviceSynchronize() );
    print_float(test, Nx, Ny);

    cuComplex* dctest;
    cuda_error_func( cudaMallocManaged( &dctest, sizeof(cuComplex)*(Ny+1)/2*Nx));
    float* dtest;
    cuda_error_func( cudaMallocManaged( &dtest, sizeof(float)*Nx*Ny));
    for (int j = 0; j < Ny/2+1; j++) {
        for (int i = 0; i < Nx; i++) {
            dctest[i*Ny+j] = im()*ky[j]*ctest[i*Ny+j];
        }
    }
    cufftExecC2R(plan,dctest,dtest);
    cout << "derivative: " << endl;
    print_float(dtest, Nx, Ny);
    return 0;
}