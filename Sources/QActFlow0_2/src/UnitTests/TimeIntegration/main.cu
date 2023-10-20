#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Field/Field.h>
#include <Basic/cuComplexBinOp.h>
#include <TimeIntegration/RK4.cuh>
#include <stdlib.h>

using namespace std;
__global__ void init_func(float* fp, float dx, float dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        fp[index] = -2.0 - sin(x+y)*sin(x+y);
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
    Field *unonl = new Field(mesh);
    Field *ucurr = new Field(mesh);
    Field *unew = new Field(mesh);
    float *IFu, *IFuh;
    cudaMallocManaged(&IFu, sizeof(float)*Nxh*Ny);
    cudaMallocManaged(&IFuh, sizeof(float)*Nxh*Ny);
    int m = 0;
    // initialize the field
    ulin_func<<<mesh->dimGridp,mesh->dimBlockp>>>(IFuh, IFu, mesh->k_squared, mesh->dt, mesh->Nxh, mesh->Ny, mesh->BSZ);
    FwdTrans(mesh, u->phys, u->spec);
    init_func<<<mesh->dimGridp,mesh->dimBlockp>>>(u->phys, 
    mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ); 
    cuda_error_func( cudaDeviceSynchronize() ); 
    field_visual(u, "u_init.csv");
    unonl_func(unonl, u);
    field_visual(unonl, "unonl_ts.csv");
    FldSet<<<mesh->dimGridp,mesh->dimBlockp>>>(unonl->phys, 0., 
    mesh->Nx, mesh->Ny, mesh->BSZ);

    m+=1;
    for(;m <Ns; m++){
        integrate_func0<<<mesh->dimGridp,mesh->dimBlockp>>>(u->spec, ucurr->spec, unew->spec, IFu, IFuh, 
        mesh->Nxh, mesh->Ny, mesh->BSZ, mesh->dt);

        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        unonl_func(unonl, ucurr);
        FwdTrans(mesh, unonl->phys, unonl->spec);

        integrate_func1<<<mesh->dimGridp,mesh->dimBlockp>>>(u->spec, ucurr->spec, unew->spec, unonl->spec, 
        IFu, IFuh, Nxh, Ny, mesh->BSZ, mesh->dt);

        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        unonl_func(unonl, ucurr);
        FwdTrans(mesh, unonl->phys, unonl->spec);

        integrate_func2<<<mesh->dimGridp,mesh->dimBlockp>>>(u->spec, ucurr->spec, unew->spec, unonl->spec, 
        IFu, IFuh, Nxh, Ny, mesh->BSZ, mesh->dt);

        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        unonl_func(unonl, ucurr);
        FwdTrans(mesh, unonl->phys, unonl->spec);

        integrate_func3<<<mesh->dimGridp,mesh->dimBlockp>>>(u->spec, ucurr->spec, unew->spec, unonl->spec, 
        IFu, IFuh, Nxh, Ny, mesh->BSZ, mesh->dt);

        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        unonl_func(unonl, ucurr);
        FwdTrans(mesh, unonl->phys, unonl->spec);

        integrate_func4<<<mesh->dimGridp,mesh->dimBlockp>>>(u->spec, ucurr->spec, unew->spec, unonl->spec, 
        IFu, IFuh, Nxh, Ny, mesh->BSZ, mesh->dt);

        BwdTrans(mesh, ucurr->spec, ucurr->phys);
        unonl_func(unonl, ucurr);
        FwdTrans(mesh, unonl->phys, unonl->spec);

        SpecSet(u->spec, unew->spec, Nxh, Ny, mesh->BSZ);
        BwdTrans(mesh, u->spec, u->phys);
        cout << u[10] << endl;
        if (m%100 == 0){
            field_visual(u, itoa(m)+"u.csv")
        }
    }


    
    return 0;
}
