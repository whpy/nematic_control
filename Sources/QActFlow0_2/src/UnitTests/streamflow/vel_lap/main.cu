#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Field/Field.h>
#include <Stream/Streamfunc.cuh>
#include <iostream>

using namespace std;

// \phi = cos(x)*sin(y)
// w = Laplacian(\phi) = -2*cos(x)*sin(y)
// u = -1*Dy(\phi) = -1*cos(x)*cos(y)
// v = Dx(\phi) = -1*sin(x)*sin(y)
__global__
void PhiinitD(float* phys, int Nx, int Ny, int BSZ, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    float x = i*dx;
    float y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = cos(x)*sin(y);
    }
}

__global__
void wexactD(float* phys, int Nx, int Ny, int BSZ, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    float x = i*dx;
    float y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = -2*cos(x)*sin(y);
    }
}

__global__
void uexactD(float* phys, int Nx, int Ny, int BSZ, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    float x = i*dx;
    float y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = -1*cos(x)*cos(y);
    }
}
__global__
void vexactD(float* phys, int Nx, int Ny, int BSZ, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    float x = i*dx;
    float y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = -1*sin(x)*sin(y);
    }
}
inline void init(Field* phi, Field* wa, Field* ua, Field* va){
    Mesh* mesh = phi->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ; 
    float dx = mesh->dx; float dy = mesh->dy;
    dim3 dimGrid = mesh->dimGridp; dim3 dimBlock = mesh->dimBlockp;
    PhiinitD<<<dimGrid, dimBlock>>>(phi->phys, Nx, Ny, BSZ, dx, dy);
    wexactD<<<dimGrid, dimBlock>>>(wa->phys, Nx, Ny, BSZ, dx, dy);
    uexactD<<<dimGrid, dimBlock>>>(ua->phys, Nx, Ny, BSZ, dx, dy);
    vexactD<<<dimGrid, dimBlock>>>(va->phys, Nx, Ny, BSZ, dx, dy);
    // update the spectral
    FwdTrans(mesh, phi->phys, phi->spec);
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

void coord(Mesh &mesh){
    ofstream xcoord("x.csv");
    ofstream ycoord("y.csv");
    for (int j=0; j<mesh.Ny; j++){
        for ( int i=0; i< mesh.Nx; i++){
            float x = mesh.dx*i;
            float y = mesh.dy*j;
            xcoord << x << ",";
            ycoord << y << ",";
        }
        xcoord << endl;
        ycoord << endl;
    }
    xcoord.close();
    ycoord.close();
}
// we test the necessary stream function method in this file 
//////////// 1////////////
// firstly the stream function is set to be \phi = cos(x)*sin(y)
// so that we could compute that:
// w = Laplacian(\phi) = -2*cos(x)*sin(y), wa
// u = -1*Dy(\phi) = -1*cos(x)*sin(y), ua
// v = Dx(\phi) = -1*sin(x)*sin(y), va
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
    float dt = 0.05; // same as colin
    float a = 1.0;

    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field* phi = new Field(mesh);
    Field* w = new Field(mesh); Field* wa = new Field(mesh);
    Field* u = new Field(mesh); Field* ua = new Field(mesh);
    Field* v = new Field(mesh); Field* va = new Field(mesh);

    coord(*mesh);
    init(phi, wa, ua, va);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(phi, "phi.csv");
    field_visual(wa, "wa.csv");
    field_visual(ua, "ua.csv");
    field_visual(va, "va.csv");
    
    // evaluate the spectral
    laplacian_func(phi->spec, w->spec, mesh);
    // switch back
    BwdTrans(mesh, w->spec, w->phys);

    // compute the velocity
    vel_func(w, u, v);
    cuda_error_func( cudaDeviceSynchronize() );

    field_visual(w, "w.csv");
    field_visual(u, "u.csv");
    field_visual(v, "v.csv");



    return 0;
}