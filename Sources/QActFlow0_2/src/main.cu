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

int main(){
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
    cuComplex b = make_cuComplex(0.,0.);
    cuComplex c = a+b;
    cout << c.x << "," << c.y <<endl;
    cout << "hello world" << endl;

    Mesh *mesh = new Mesh(Nx, Ny, Lx, Ly);
    Field *u = new Field(mesh);
    FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(u->phys, 1., mesh->Nx, mesh->Ny, mesh->BSZ);
    field_visual(u ,"u.csv");
    return 0;
}
