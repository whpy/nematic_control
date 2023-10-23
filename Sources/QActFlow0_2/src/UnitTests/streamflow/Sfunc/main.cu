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
void rinitD(float* r1, float* r2, int Nx, int Ny, int BSZ, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    float x = i*dx;
    float y = j*dy;
    if (i<Nx && j<Ny){
        r1[index] = cos(x+y);
        r2[index] = sin(x+y);
    }
}

// S = 2*sqrt(r1^2+r2^2) = 2
__global__
void SexactD(float* S, int Nx, int Ny, int BSZ, float dx, float dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    float x = i*dx;
    float y = j*dy;
    if (i<Nx && j<Ny){
        S[index] = 2.f;
    }
}
// __global__
// void wexactD(float* phys, int Nx, int Ny, int BSZ, float dx, float dy){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = i + j*Nx;
//     float x = i*dx;
//     float y = j*dy;
//     if (i<Nx && j<Ny){
//         phys[index] = -2*cos(x)*sin(y);
//     }
// }

// __global__
// void uexactD(float* phys, int Nx, int Ny, int BSZ, float dx, float dy){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = i + j*Nx;
//     float x = i*dx;
//     float y = j*dy;
//     if (i<Nx && j<Ny){
//         phys[index] = -1*cos(x)*cos(y);
//     }
// }
// __global__
// void vexactD(float* phys, int Nx, int Ny, int BSZ, float dx, float dy){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = i + j*Nx;
//     float x = i*dx;
//     float y = j*dy;
//     if (i<Nx && j<Ny){
//         phys[index] = -1*sin(x)*sin(y);
//     }
// }
inline void init(Field* r1, Field* r2, Field* Sa){
    Mesh* mesh = r1->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ; 
    float dx = mesh->dx; float dy = mesh->dy;
    dim3 dimGrid = mesh->dimGridp; dim3 dimBlock = mesh->dimBlockp;
    rinitD<<<dimGrid, dimBlock>>>(r1->phys, r2->phys, Nx, Ny, BSZ, dx, dy);
    SexactD<<<dimGrid, dimBlock>>>(Sa->phys, Nx, Ny, BSZ, dx, dy);
    // uexactD<<<dimGrid, dimBlock>>>(ua->phys, Nx, Ny, BSZ, dx, dy);
    // vexactD<<<dimGrid, dimBlock>>>(va->phys, Nx, Ny, BSZ, dx, dy);
    // update the spectral
    // FwdTrans(mesh, r1->phys, ->spec);
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

/////////// 2 ////////////
// we previously have verified the validity of laplacian and vel_func.
// in this file we test the func about the Q tensor (components, r1, r2) and 
// the intermediate components p (p11, p12, p21)
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field *r1 = new Field(mesh); Field *r2 = new Field(mesh);
    Field *S = new Field(mesh); Field *Sa = new Field(mesh);

    // aux is the abbreviation of auxiliary, where only act as intermediate values
    // to assist computation. So we should guarantee that it doesnt undertake any 
    // long term memory work.
    Field *aux = new Field(mesh); Field *aux1 = new Field(mesh);
    // Field* phi = new Field(mesh);
    // Field* w = new Field(mesh); Field* wa = new Field(mesh);
    // Field* u = new Field(mesh); Field* ua = new Field(mesh);
    // Field* v = new Field(mesh); Field* va = new Field(mesh);

    coord(*mesh);
    init(r1, r2, Sa);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(r1, "r1.csv");
    field_visual(r2, "r2.csv");
    field_visual(Sa, "Sa.csv");
    
    // evaluate the S
    S_func(r1, r2, S);
    // S_funcD<<<mesh->dimGridp, mesh->dimBlockp>>>(r1->phys, r2->phys, S->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(S, "S.csv");
//////////////////// Sfunc tested //////////////////

//////////////////// crossfunc test ///////////////









    return 0;
}