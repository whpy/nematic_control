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
        fp[index] = cos((x+y));
    }
}

__global__
void symtest_func(cuComplex* fsp, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        if (j<Ny/2){
            fsp[index] = make_cuComplex(-1.f,1.f);
        }
        else{
            fsp[index] = make_cuComplex(2.f,-2.f);
        }
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

__global__ void gen_func(float* fp, float dx, float dy, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        float x = i*dx;
        float y = j*dy;
        fp[index] = exp(-1*((x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI)));
    }
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

int main(){
    int BSZ = 16;
    int Ns = 1000;
    int Nx = 2048*3/2; // same as colin
    int Ny = 2048*3/2;
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
    coord(*mesh);

    symtest_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(u->spec, Nxh, Ny, BSZ);
    FwdTrans(mesh, u->phys, u->spec);
    cout << "before symmetry" << endl;
    cuda_error_func( cudaDeviceSynchronize() );
    // print_spec(u);

    
    symmetry_func<<<mesh->dimGridp, mesh->dimBlockp>>>(u->spec, Nxh, Ny, BSZ);
    cout << "after symmetry" << endl;
    cuda_error_func( cudaDeviceSynchronize() );

    
    // print_spec(u);
    cout << endl;
    printf("sp: Grid:%d, %d Block: %d, %d \n", mesh->dimGridsp.x, mesh->dimGridsp.y, mesh->dimBlocksp.x, mesh->dimBlocksp.y);
    printf("p: Grid:%d, %d Block: %d, %d \n", mesh->dimGridp.x, mesh->dimGridp.y, mesh->dimBlockp.x, mesh->dimBlockp.y);
    
    cout << "b4 dealiasing" << endl;
    cuda_error_func( cudaDeviceSynchronize() );
    // print_spec(u);
    dealiasing_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(u->spec, mesh->cutoff, Nxh, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << "after dealiasing" << endl;
    // print_spec(u);

    gen_func<<<mesh->dimGridp, mesh->dimBlockp>>>(w->phys, dx, dy, Nx, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(w,"benchmark.csv");
    FwdTrans(mesh, w->phys, v->spec);
    BwdTrans(mesh, v->spec, v->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(v,"single.csv");

    
    FwdTrans(mesh, w->phys, u->spec);
    // symmetry_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(u->spec, Nxh, Ny, BSZ);
    // dealiasing_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(u->spec, u->mesh->cutoff, Nxh, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    cout << endl;
    // print_spec(u);
    BwdTrans(mesh, u->spec, u->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(u,"modifiedTrans.csv");
    
    return 0;
}
