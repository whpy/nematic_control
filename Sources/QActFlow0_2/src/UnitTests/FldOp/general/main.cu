#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Field/Field.h>
#include <Basic/cuComplexBinOp.h>

using namespace std;

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

__global__ void cos_func(float* fp, float dx, float dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        fp[index] = cos(x+y);
    }
}

__global__ void sin_func(float* fp, float dx, float dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        fp[index] = sin(x+y);
    }
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
    Field *v = new Field(mesh);
    Field *w = new Field(mesh);

    FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(u->phys, 1., mesh->Nx, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(u ,"u1.csv");

    // FldAdd in place test
    FldAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(2., u->phys, -0.5, u->phys, u->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(v->phys, u->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(v ,"v1.csv");
    field_visual(u ,"u2.csv");

    //FldMul in place test
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(u->phys, v->phys, 3.5, u->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(u ,"u3.csv");

    // Fwd and Bwd test
    // from u.phys to v.spec then bwd to v.phys
    FwdTrans(mesh, u->phys, v->spec);
    BwdTrans(mesh, v->spec, v->phys);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(v ,"v2.csv");

    //specset inplace test
    SpecSet<<<mesh->dimGridp, mesh->dimBlockp>>>(v->spec, make_cuComplex(0., 0.), mesh->Nxh, mesh->Ny, mesh->BSZ);
    BwdTrans(mesh, v->spec, v->phys);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(v ,"v3.csv");

    //specadd inplace test
    // v = -u should be
    FwdTrans(mesh, u->phys, u->spec);
    SpecAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(1., v->spec, -1., u->spec, v->spec, mesh->Nxh, mesh->Ny, mesh->BSZ); 
    BwdTrans(mesh, v->spec, v->phys);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(v ,"v4.csv");

    // derivatives test
    cos_func<<<mesh->dimGridp, mesh->dimBlockp>>>(u->phys, mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ);
    sin_func<<<mesh->dimGridp, mesh->dimBlockp>>>(v->phys, mesh->dx, mesh->dy, mesh->Nx, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(v ,"v5.csv");
    field_visual(u ,"u4.csv");
    // a small trick to solve a = b*b + c*c, without affecting the physical space,
    // key is to utilize that hat(a) + hat(b) = hat(a+b), 
    // hence we could largely use the temporarily unused spectral memory. 
    // Here the w = u*u + v*v = 1
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(u->phys, u->phys, 1., w->phys, w->mesh->Nx, w->mesh->Ny, w->mesh->BSZ);
    FwdTrans(w->mesh, w->phys, u->spec);
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(v->phys, v->phys, 1., w->phys, w->mesh->Nx, w->mesh->Ny, w->mesh->BSZ);
    FwdTrans(w->mesh, w->phys, w->spec);
    SpecAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(1., u->spec, 1., w->spec, w->spec, w->mesh->Nxh, w->mesh->Ny, w->mesh->BSZ);
    BwdTrans(w->mesh, w->spec, w->phys);
    cuda_error_func( cudaDeviceSynchronize());
    // w = u*u + v*v = 1
    field_visual(w ,"w1.csv");

    // the spectral of u has been polluted before
    FwdTrans(u->mesh, u->phys, u->spec);
    xDeriv(u->spec, w->spec, w->mesh);
    BwdTrans(w->mesh, w->spec, w->phys);
    FldAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(1., w->phys, 1., v->phys, w->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize());
    // w = 0.
    field_visual(w ,"w2.csv");

    FwdTrans(v->mesh, v->phys, v->spec);
    yDeriv(v->spec, w->spec, w->mesh);
    BwdTrans(w->mesh, w->spec, w->phys);
    FldAdd<<<mesh->dimGridp, mesh->dimBlockp>>>(1., w->phys, -1., u->phys, w->phys, mesh->Nx, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaDeviceSynchronize());
    // w = 0.
    field_visual(w ,"w3.csv");
    return 0;
}
