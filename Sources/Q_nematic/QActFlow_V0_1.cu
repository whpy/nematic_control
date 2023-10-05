#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <chrono>
#include <random>

#include <cufft.h>
#include <cuComplex.h>
#include <cuComplexBinOp.h>
#include <cudaErr.h>

#define _pi 3.14159265358979323846
// amount of kernels called, block_size must be n*32
#define BLOCK_SIZE 1024
#define BLOCK_NUM 64


using namespace std;

/******************************************************************************* 
 * global variables                                                            * 
 *******************************************************************************/

// number of time steps, length of one time step, total time
int Ns{5000};
float dt{ 0.002 };
float T = Ns*dt;
//non-dimensional number 
float lambda{};
float C_cn{};
float C_cf{};
float Pe{};
float Re{};
// total length, number of gridpoints and array size
float Lx{ 250 };
float Ly{ Lx };
int Nx{ 1024 };
int Ny{ Nx };

/******************************************************************************* 
 *  variables to be solved                                                           * 
 *******************************************************************************/
// the most principle value we concern
float *w, *r1, *r2;
// derived quantities that solving involved 
float *u, *v, *S;
// the manipulate term, \Tiled{alpha}
float *Talpha;

//the wave number field
float *kx, *ky;
// integrating factors, unit and half unit in RK4 
float *wlin, *wlin2;
float *r1lin, *r1lin2;
float *r2lin, *r2lin2;

/******************************************************************************* 
 * prototypes of functions evolved                                                                * 
 *******************************************************************************/

// compute the cross terms in solving w_z 
__global__ void cal_cross(cuComplex* p11, cuComplex* p12, cuComplex* p21);

// the non-linear term of corresponding variable
__global__ void r1NL();
__global__ void r2NL();
__global__ void wNL();

// the integrating factors of corresponding variable
__global__ void r1lin_func();
__global__ void r2lin_func();
__global__ void wlin_func();



void initialize(){
    cuda_error_func(cudaMallocManaged( &kx, sizeof(float)*Nx));
    cuda_error_func(cudaMallocManaged( &ky, sizeof(float)*Ny));
}
int main(){
    initialize();
    cout << kx[0] << endl;
    cout << "hello world!" << endl;
    return 0;
}