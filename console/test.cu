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

class session{
    public:
        int Nx;
        int Ny;

        float Lx;
        float dx;
        float alphax;

        float Ly;
        float dy;
        float alphay;

        session(int Nx, int Ny, float Lx, float Ly):Nx(Nx), Ny(Ny), Lx(Lx),Ly(Ly){
            dx = Lx/Nx;
            alphax = Lx/
            dy = Ly/Ny;

        }

};
//cufft_error_func( cufftPlan2d( &transf, Ny, Nx, CUFFT_R2C ) );
//cufft_error_func( cufftPlan2d( &inv_transf, Ny, Nx, CUFFT_C2R ) );

__global__ void D_init(float *t, int Nx, int Ny, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i < Nx && j < Ny){
        t[index] = sin((float)j*dy);
    }
}

__global__ void exact(float *t, int Nx, int Ny, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
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
            cout <<t[j*Nx+i] << ",";
        }
        cout << endl;
    }
}

void transforwards(float *f, cuComplex *ft)
{
    cufftHandle plan;
    cufftPlan2d(&plan, Nx, Ny,CUFFT_C2C);
    cufftExecC2C(plan, f, ft, CUFFT_FORWARD);
} 

void transbackwards(cuComplex *ft, float *f){
    cufftHandle plan;
    cufftPlan2d(&plan, Nx, Ny,CUFFT_C2C);
    cufftExecC2C(plan, ft, f, CUFFT_INVERSE);
    ComplextoReal(f)
}
__global__ void Dx(cuComplex *ft, cuComplex *dft, float* kx, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;

    dft[index] = ft[index]*im()*kx[i];
}

__global__ void Dy(cuComplex *ft, cuComplex *dft, float* ky, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;

    dft[index] = ft[index]*im()*ky[j];
}

__global__ void RealtoComplex(float *f, cuComplex *fc, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    fc[index].x = f[index];
    fc[index].y = 0;
}

__global__ void ComplextoReal(cuComplex *fc, float *f, int Nx, int Ny){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    f[index] = fc[index].x/(Nx*Ny);
}

int main(){
    float *test;
    float *exe;
    cuComplex* ctest;
    cuComplex *htest;
    cuComplex *tmp;
    float *kx;
    float *ky;
    cuda_error_func(cudaMallocManaged( &test, sizeof(float)*(Nx*Ny) ) );
    cuda_error_func(cudaMallocManaged( &exe, sizeof(float)*(Nx*Ny) ) );
    cuda_error_func(cudaMallocManaged( &ctest, sizeof(cuComplex)*(Nx*Ny)));
    cuda_error_func(cudaMallocManaged( &htest, sizeof(cuComplex)*(Nx*Ny)));
    cuda_error_func(cudaMallocManaged( &tmp, sizeof(cuComplex)*(Nx*Ny)));
    cuda_error_func(cudaMallocManaged( &kx, sizeof(float)*(Nx)));
    cuda_error_func(cudaMallocManaged( &ky, sizeof(float)*Ny));
    float alpha = 2*M_PI;
<<<<<<< HEAD
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
=======
    for (int i=0; i<=Nx/2; i++)          
	{
	   kx[i] = i*2*M_PI/alpha;
    } 
	for (int i=Nx/2+1; i<Nx; i++)          
	{
        kx[i] = (i - Nx) * 2*M_PI/alpha;
	}

    for (int i=0; i<=Ny/2; i++)          
	{
	   ky[i] = i*2*M_PI/alpha;
    } 
	for (int i=Ny/2+1; i<Ny; i++)          
	{
        ky[i] = (i - Ny) * 2*M_PI/alpha;
	}

>>>>>>> d327c79163cf810f7a7f6bf92388042ffd7b3676
    init(test);
    exact<<<dimGrid,dimBlock>>>(exe, Nx, Ny, dx, dy);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << "exact" << endl;
    print_float(exe, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << "input: " << endl;
    print_float(test, Nx, Ny);
    RealtoComplex<<<dimGrid,dimBlock>>>(test, ctest, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << "here" << endl;

    cufftHandle plan;
    cufftPlan2d(&plan, Nx, Ny,CUFFT_C2C);
    cufftExecC2C(plan, ctest, htest, CUFFT_FORWARD);
    cufftExecC2C(plan, htest, ctest, CUFFT_INVERSE);
    ComplextoReal<<<dimGrid,dimBlock>>>(ctest, test, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << endl << "after fft and ifft" << endl;
    print_float(test, Nx, Ny);

    Dx<<<dimGrid,dimBlock>>>(htest, tmp, kx, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    cufftExecC2C(plan,tmp, ctest, CUFFT_INVERSE);
    ComplextoReal<<<dimGrid,dimBlock>>>(ctest, test, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << endl << "after derive x" << endl;
    print_float(test, Nx, Ny);

    Dy<<<dimGrid,dimBlock>>>(htest, tmp, kx, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    cufftExecC2C(plan,tmp, ctest, CUFFT_INVERSE);
    ComplextoReal<<<dimGrid,dimBlock>>>(ctest, test, Nx, Ny);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << endl << "after derive y" << endl;
    print_float(test, Nx, Ny);

    return 0;
}