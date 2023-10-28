#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Field/Field.h>
#include <Basic/cuComplexBinOp.h>

using namespace std;
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

__global__ void u_func(float* fp, float dx, float dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        fp[index] = cos(x+y);
    }
}

__global__ void Lua_func(float* fp, float dx, float dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        fp[index] = -2.f*cos(x+y);
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
    int Nx = 512*3/2; // same as colin
    int Ny = 512*3/2;
    int Nxh = Nx/2+1;
    float Lx = 2*M_PI;
    float Ly = 2*M_PI;
    float dx = 2*M_PI/Nx;
    float dy = 2*M_PI/Ny;
    float dt = 0.005; // same as colin
    float a = 1.0;

    // Fldset test
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field *u = new Field(mesh); Field *v = new Field(mesh);
    Field *Lu = new Field(mesh); Field *Lua = new Field(mesh);
    Field *aux = new Field(mesh);
    coord(*mesh);
    u_func<<<mesh->dimGridp, mesh->dimBlockp>>>(u->phys, dx, dy, Nx, Ny,BSZ);
    FwdTrans(mesh, u->phys, u->spec);
    u_func<<<mesh->dimGridp, mesh->dimBlockp>>>(v->phys, dx, dy, Nx, Ny,BSZ);
    FwdTrans(mesh, v->phys, v->spec);
    Lua_func<<<mesh->dimGridp, mesh->dimBlockp>>>(Lua->phys, dx, dy, Nx, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    // print_spec(u);
    field_visual(Lua, "Lua.csv");
    field_visual(u, "u.csv");
    laplacian_func(u->spec, Lu->spec, mesh);
    // dealiasing_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(Lu->spec, mesh->cutoff, Nxh, Ny, BSZ);
    // symmetry_func<<<mesh->dimGridsp,mesh->dimBlocksp>>>(Lu->spec, Nxh, Ny, BSZ);
    BwdTrans(mesh, Lu->spec, Lu->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(Lu, "Lu.csv");

    








    

    
    return 0;
}
