#include <iostream>
#include <fstream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>
#include <stdio.h>
#include "cuComplexBinOp.h"
#include "cudaErr.h"

#include <cufft.h>
#include <cuComplex.h>
#include "cuComplexBinOp.h"
#define BSZ 4

using namespace std;

__global__ 
void hello1D(){
    printf("hello1 \n");
}

__global__ 
void hello2D(){
    printf("hello2 \n");
}

void hello1(){
    hello1D<<<1,16>>>();
}

void hello2(){
    hello2D<<<1,32>>>();
}

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

void print_func(cuComplex *spec, int Nxh, int Ny){
    for (int j = 0; j < Ny; j++){
        for (int i = 0; i < Nxh; i++){
            int index = j*Nxh + i;
            printf("( %f, %f), ", spec[index].x, spec[index].y);
        }
        printf("\n");
    }
}
int main(){
    cuComplex *spec;
    
    int Ny = 16;
    int Nx = 16;
    int Nxh = Nx/2+1;
    dim3 dimGrid  (int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dim3 dimBlock (BSZ, BSZ); 

    cudaMallocManaged(&spec, sizeof(cuComplex)*Ny*Nxh);
    for (int i = 0; i < Nxh; i++){
        for (int j = 0; j < Ny; j++){
            int index = j*Nxh + i;
            if (i==0){
                spec[index] = make_cuComplex((float)(j)/2, (float)j);
            }
            else{
                spec[index] = make_cuComplex(0.f,0.f);
            }
        }
    }
    print_func(spec, Nxh, Ny);
    cout << "after symmetry." <<endl;
    reality_func<<<dimGrid, dimBlock>>>(spec, Nxh, Ny);
    cudaDeviceSynchronize();
    print_func(spec, Nxh, Ny);
    
    return 0;
}