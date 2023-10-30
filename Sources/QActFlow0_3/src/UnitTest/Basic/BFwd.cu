#include <Basic/QActFlow.h>
#include <Basic/FldOp.hpp>
#include <Basic/Field.h>
#include <Basic/cuComplexBinOp.hpp>

using namespace std;

__global__ void sin_func(Qreal* fp, Qreal dx, Qreal dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        Qreal x = i*dx;
        Qreal y = j*dy;
        fp[index] = sin(x+y);
    }
}

void coord(Mesh &mesh){
    ofstream xcoord("x.csv");
    ofstream ycoord("y.csv");
    for (int j=0; j<mesh.Ny; j++){
        for ( int i=0; i< mesh.Nx; i++){
            double x = mesh.dx*i;
            double y = mesh.dy*j;
            xcoord << x << ",";
            ycoord << y << ",";
        }
        xcoord << endl;
        ycoord << endl;
    }
    xcoord.close();
    ycoord.close();
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
int main(){
    int BSZ = 16;
    int Ns = 1000;
    int Nx = 512*3/2; // same as colin
    int Ny = 512*3/2;
    int Nxh = Nx/2+1;
    Qreal Lx = 2*M_PI;
    Qreal Ly = 2*M_PI;
    Qreal dx = 2*M_PI/Nx;
    Qreal dy = 2*M_PI/Ny;
    Qreal dt = 0.005; // same as colin
    Qreal a = 1.0;

    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    coord(*mesh);
    Field *u = new Field(mesh);
    Field *v = new Field(mesh);
    Field *w = new Field(mesh);
    sin_func<<<mesh->dimGridp, mesh->dimBlockp>>>(u->phys, dx, dy, Nx, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(u ,"u1.csv");

    FwdTrans(u->mesh, u->phys, u->spec);
    BwdTrans(u->mesh, u->spec, u->phys);
    cuda_error_func( cudaDeviceSynchronize());
    field_visual(u ,"u2.csv");


    return 0;
}
