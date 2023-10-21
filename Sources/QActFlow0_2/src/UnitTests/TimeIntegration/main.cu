#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Field/Field.h>
#include <Basic/cuComplexBinOp.h>
#include <TimeIntegration/RK4.cuh>
#include <stdlib.h>
#include <iostream>

using namespace std;
__global__ void init_func(float* fp, float dx, float dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        fp[index] = -sin(x+y);
    }
}

void field_visual(Field *f, string name){
    Mesh* mesh = f->mesh;
    ofstream fval;
    string fname = name;
    fval.open(fname);
    for (int j=0; j<mesh->Ny; j++){
        for (int i=0; i<mesh->Nx; i++){
            fval << f->phys[j*mesh->Nx+i] << ",";
        }
        fval << endl;
    }
    fval.close();
}

// du/dt = u^2+2*2+1 = L(u) + NL(u),
// L(u) = 2*u, NL(u) = u^2+1
__global__
void ulin_func(float* IFuh, float* IFu, float* k_squared, 
float dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    float alpha = 2.0f;
    if(i<Nxh && j<Ny){
        IFuh[index] = exp( alpha *dt/2);
        IFu[index] = exp( alpha *dt);
    }
}

void unonl_func(Field* unonl, Field* ucurr,float t){
    Mesh* mesh = unonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    // unonl = ucurr*ucurr
    FldMul<<<dimGrid, dimBlock>>> (ucurr->phys, ucurr->phys, 1.0, unonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
    // unonl = unonl + 1 = ucurr*ucurr + 1
    cuda_error_func( cudaDeviceSynchronize() );
    FldAdd<<<dimGrid, dimBlock>>> (1., unonl->phys, 1.0, unonl->phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    FwdTrans(unonl->mesh, unonl->phys, unonl->spec);
}

void print_spec(Field* f){
    Mesh* mesh = f->mesh;
    int Nxh = mesh->Nxh, Ny = mesh->Ny;
    for(int j = 0; j < Ny; j++){
        for (int i = 0; i < Nxh; i++){
            int index = i + j*Nxh;
            cout << "("<< f->spec[index].x << "," << f->spec[index].y << ")" << " ";
        }
        cout << endl;
    }
}

void print_phys(Field* f){
    Mesh* mesh = f->mesh;
    int Nx = mesh->Nx, Ny = mesh->Ny;
    for(int j = 0; j < Ny; j++){
        for (int i = 0; i < Nx; i++){
            int index = i + j*Nx;
            cout << "("<< f->phys[index]<< ")" << " ";
        }
        cout << endl;
    }
}
__global__ 
void tintegrate_func1(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
                    float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt){
    // spec_nonl = a_n/dt here
    // spec_curr represents the value to be input into Nonlinear function for b_n/dt next 
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        
        cuComplex an = spec_nonl[index]*dt;
        if(i == 0 && j == 0){
            printf("dt = %f \n",dt);
            printf("an = (%f, %f)\n", an.x, an.y);
            printf("IF[index] = %f\n", IF[index]);
            printf("1/6*IF[index] = %f\n", 1/6*IF[index]);
            printf("1/6 = %f\n", 1/6);
            printf("1./6. = %f\n", 1./6.);
        }
        // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n)
        spec_new[index] = spec_new[index] + 1./6.*IF[index] * an;
        // (u_{n}+a_{n}/2)*exp(alpha*dt/2)
        spec_curr[index] = (spec_old[index]+an/2) * IFh[index];
    }
}

int main(){
    int BSZ = 16;
    int Ns = 1000;
    int Nx = 8; // same as colin
    int Ny = 8;
    int Nxh = Nx/2+1;
    float Lx = 2*M_PI;
    float Ly = 2*M_PI;
    float dx = 2*M_PI/Nx;
    float dy = 2*M_PI/Ny;
    float dt = 0.05; // same as colin
    float a = 1.0;

    // Fldset test
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field *u = new Field(mesh);
    Field *unonl = new Field(mesh);
    Field *ucurr = new Field(mesh);
    Field *unew = new Field(mesh);
    float *IFu, *IFuh;
    cudaMallocManaged(&IFu, sizeof(float)*Nxh*Ny);
    cudaMallocManaged(&IFuh, sizeof(float)*Nxh*Ny);
    int m = 0;
    // initialize the field
    // set up the Integrating factor
    // we may take place here by IF class
    ulin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFuh, IFu, mesh->k_squared, dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // initialize the physical space of u(u_o.x << "," << f->phys[index].y ld)
    init_func<<<mesh->dimGridp,mesh->dimBlockp>>>(u->phys, 
    mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ);
    // initialize the spectral space of u 
    FwdTrans(mesh, u->phys, u->spec);
    cuda_error_func( cudaDeviceSynchronize() );
    for (int j = 0; j <Ny; j++){
        for (int i = 0; i < Nxh; i++){
            int index = j*Nxh + i;
            cout << IFu[index] << " ";
        }
        cout << endl;
    }
    cout << endl;
    for (int j = 0; j <Ny; j++){
        for (int i = 0; i < Nxh; i++){
            int index = j*Nxh + i;
            cout << IFuh[index] << " ";
        }
        cout << endl;
    }

    SpecSet<<<mesh->dimGridsp,mesh->dimBlocksp>>>(unonl->spec, make_cuComplex(1.f,0.f), Nxh, Ny, BSZ);
    SpecSet<<<mesh->dimGridsp,mesh->dimBlocksp>>>(u->spec, make_cuComplex(0.f,1.f), Nxh, Ny, BSZ);
    SpecSet<<<mesh->dimGridsp,mesh->dimBlocksp>>>(unew->spec, make_cuComplex(0.f,2.f), Nxh, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << "b4 func1" << endl;
    print_spec(unew);
    tintegrate_func1<<<mesh->dimGridsp,mesh->dimBlocksp>>>(u->spec, ucurr->spec, unew->spec, unonl->spec, IFu, IFuh,Nxh, Ny, BSZ, dt);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << "afer func1" << endl;
    print_spec(unew);




    return 0;
}