#include <Basic/QActFlow.h>
#include <Field/Field.h>
#include <Basic/Mesh.h>

using namespace std;
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

void field_visual(Field &f, string name){
    Mesh* mesh = f.mesh;
    ofstream fval;
    string fname = name;
    fval.open(fname);
    for (int j=0; j<mesh->Ny; j++){
        for (int i=0; i<mesh->Nx; i++){
            fval << f.phys[j*mesh->Nx+i] << ",";
        }
        fval << endl;
    }
    fval.close();
}
void print_float(float* t, int Nx, int Ny) {
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            cout <<t[j*Nx+i] << ",";
        }
        cout << endl;
    }
}

void print_spec(cuComplex* t, int Nxh, int Ny) {
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nxh; i++) {
            cout <<t[j*Nxh+i].x<<","<<t[j*Nxh+i].y << "  ";
        }
        cout << endl;
    }
}