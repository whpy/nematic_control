#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Field/Field.h>
#include <Basic/cuComplexBinOp.h>
#include <TimeIntegration/RK4.cuh>

using namespace std;
__global__ void init_func(float* fp, float dx, float dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        fp[index] = -1.2 - sin(x+y)*sin(x+y);
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
void ulin_func(float* IFuh, float* IFu, float* k_squared, float Re, float cf, float dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    float alpha = 2.0;
    if(i<Nxh && j<Ny){
        IFuh[index] = exp( alpha *dt/2);
        IFu[index] = exp( alpha *dt);
    }
}

void unonl_func(Field unonl, Field ucurr){
    Mesh* mesh = unonl.mesh;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    // unonl = ucurr*ucurr
    FldMul<<<dimGrid, dimBlock>>> (ucurr.phys, ucurr.phys, 1.0, unonl.phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
    // unonl = unonl + 1 = ucurr*ucurr + 1
    FldAdd<<<dimGrid, dimBlock>>> (1., unonl.phys, 1.0, unonl.phys, 
    mesh->Nx, mesh->Ny, mesh->BSZ);
}

int main(){
    int BSZ = 16;
    int Ns = 1000;
    int Nx = 512; // same as colin
    int Ny = 512;
    int Nxh = Nx/2+1;
    float Lx = 2*M_PI;
    float Ly = 2*M_PI;
    float dx = 2*M_PI/Nx;
    float dy = 2*M_PI/Ny;
    float dt = 0.005; // same as colin
    float a = 1.0;

    // Fldset test
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field *u = new Field(mesh);
    Field *ucurr = new Field(mesh);
    Field *unew = new Field(mesh);

    int m = 0;
    // initialize the field
    init_func<<<mesh->dimGridp,mesh->dimBlockp>>>(u->phys, 
    mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ);    


    
    return 0;
}
