#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Field/Field.h>
#include <Stream/Streamfunc.cuh>
#include <iostream>

using namespace std;

// \phi = cos(x)*sin(y)
// w = Laplacian(\phi) = -2*cos(x)*sin(y)
// u = -1*Dy(\phi) = -1*cos(x)*sin(y)
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
inline void phiinit(Field* phi){
    Mesh* mesh = phi->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ; 
    float dx = mesh->dx; float dy = mesh->dy;
    dim3 dimGrid = mesh->dimGridp; dim3 dimBlock = mesh->dimBlockp;
    PhiinitD<<<dimGrid, dimBlock>>>(phi->phys, Nx, Ny, BSZ, dx, dy);
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
    int Nx = 8; // same as colin
    int Ny = 8;
    int Nxh = Nx/2+1;
    float Lx = 2*M_PI;
    float Ly = 2*M_PI;
    float dx = 2*M_PI/Nx;
    float dy = 2*M_PI/Ny;
    float dt = 0.05; // same as colin
    float a = 1.0;

    return 0;
}