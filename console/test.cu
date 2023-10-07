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
int Nx = 8;
int Ny = 8;
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
    for (int i=0; i < Nx; i++) {
        kx[i] = 2*M_PI * i;
    }
    for (int j=0; j < Ny/2+1; j++) {
        if(j < Ny/2+1){
            ky[j] = 2*M_PI * j;
        }
        else{
            ky[j] = 2*M_PI * (j-Ny);
        }
    }
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