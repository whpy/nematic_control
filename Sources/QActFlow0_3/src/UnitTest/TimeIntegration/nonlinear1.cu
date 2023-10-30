#include <Basic/QActFlow.h>
#include <Basic/FldOp.hpp>
#include <Basic/Field.h>
#include <Basic/cuComplexBinOp.hpp>
#include <TimeIntegration/RK4.cuh>
#include <stdlib.h>
#include <iostream>

using namespace std;
__global__ void init_func(Qreal* fp, Qreal dx, Qreal dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        Qreal x = i*dx;
        Qreal y = j*dy;
        fp[index] = -2.0;
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

Qreal exact(Qreal t){
    return -1.0 - 1.0/(t+1.0);
}
// du/dt = u^2 + 2*u + 1 = L(u) + NL(u),
// L(u) = 2*u, NL(u) = u^2+1
__global__
void ulin_func(Qreal* IFuh, Qreal* IFu, Qreal* k_squared, 
Qreal dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    Qreal alpha = 2.0f;
    if(i<Nxh && j<Ny){
        IFuh[index] = exp( alpha *dt/2);
        IFu[index] = exp( alpha *dt);
    }
}

void unonl_func(Field* unonl, Field* ucurr,Qreal t){
    Mesh* mesh = unonl->mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    // ucurr = ucurr*ucurr
    FldMul<<<dimGrid, dimBlock>>>(ucurr->phys, ucurr->phys, 1.f, ucurr->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // ucurr = ucurr + 1 = ucurr*ucurr + 1
    FldAdd<<<dimGrid, dimBlock>>>(1.f, ucurr->phys, 1.f, ucurr->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    // unonl = ucurr = ucurr*ucurr + 1
    FldSet<<<dimGrid, dimBlock>>>(unonl->phys, ucurr->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    
    // update the spectral space
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

// we test the performance of the RK4 on linear ODE that du/dt = -u where
// the exact solution should be u = c0*exp(-t), c0 depends on initial conditon.
int main(){
    int BSZ = 16;
    int Ns = 5000000;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    Qreal Lx = 2*M_PI;
    Qreal Ly = 2*M_PI;
    Qreal dx = 2*M_PI/Nx;
    Qreal dy = 2*M_PI/Ny;
    Qreal dt = 0.005; // same as colin
    Qreal a = 1.0;

    // Fldset test
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field *u = new Field(mesh);
    Field *unonl = new Field(mesh);
    Field *ucurr = new Field(mesh);
    Field *unew = new Field(mesh);
    Qreal *IFu, *IFuh;
    cudaMallocManaged(&IFu, sizeof(Qreal)*Nxh*Ny);
    cudaMallocManaged(&IFuh, sizeof(Qreal)*Nxh*Ny);
    // int m = 0;
    // initialize the field
    // set up the Integrating factor
    // we may take place here by IF class
    ulin_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(IFuh, IFu, mesh->k_squared, dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    // initialize the physical space of u(u_o.x << "," << f->phys[index].y ld)
    init_func<<<mesh->dimGridp,mesh->dimBlockp>>>(u->phys, 
    mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ);
    // initialize the spectral space of u 
    FwdTrans(mesh, u->phys, u->spec);
    // cuda_error_func( cudaDeviceSynchronize() );
    // print_phys(u);

    // cout << "b4 nonl the unonl"<<endl;
    // // print_phys(unonl);
    // unonl_func(unonl, u, 0.f);
    // cuda_error_func( cudaDeviceSynchronize() );
    // cout << "after nonl the unonl"<<endl;
    // print_phys(unonl);

    
    for(int m=0 ;m<Ns ;m++){
        integrate_func0(u, ucurr, unew, IFu, IFuh, dt);
        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        
        unonl_func(unonl, ucurr, m*dt);
        cuda_error_func( cudaDeviceSynchronize() );
        // printf("(%f, %f)\n", unonl->spec[5].x, unonl->spec[5].y);
        integrate_func1(u, ucurr, unew, unonl, IFu, IFuh, dt);
        BwdTrans(mesh, ucurr->spec, ucurr->phys);

        unonl_func(unonl, ucurr, m*dt);
        integrate_func2(u, ucurr, unew, unonl, IFu, IFuh, dt);
        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        
        unonl_func(unonl, ucurr, m*dt);
        integrate_func3(u, ucurr, unew, unonl, IFu, IFuh, dt);
        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        
        unonl_func(unonl, ucurr, m*dt);
        integrate_func4(u, ucurr, unew, unonl, IFu, IFuh, dt);
        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        cuda_error_func( cudaDeviceSynchronize() );
        unonl_func(unonl, ucurr, m*dt);
        SpecSet<<<mesh->dimGridsp, mesh->dimBlocksp>>>(u->spec, unew->spec, mesh->Nxh, mesh->Ny, mesh->BSZ);
        
        if (m%200 == 0){
            BwdTrans(mesh, u->spec, u->phys);
            cuda_error_func( cudaDeviceSynchronize() );
            printf("t: %f    val:%.8f   exa:%.8f    err: %.8f \n",m*dt,  u->phys[5],exact((m)*dt), u->phys[5]-exact((m)*dt));
            // cout<<"t: " << m*dt << "  " << u->phys[5] << endl;
        }
    }
    
    return 0;
}
