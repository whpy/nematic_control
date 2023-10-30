#include <Basic/QActFlow.h>
#include <Basic/FldOp.cuh>
#include <Basic/Field.h>
#include <Stream/Streamfunc.cuh>
#include <iostream>

using namespace std;

// \phi = cos(3x)*sin(4y)
// w = Laplacian(\phi) = -25*cos(3x)*sin(4y)
// u = -1*Dy(\phi) = -4*cos(3x)*cos(4y)
// v = Dx(\phi) = -3*sin(3x)*sin(4y)

__global__
void PhiinitD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = sin(x+y);
    }
}

__global__
void rinitD(Qreal* r1, Qreal* r2, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        r1[index] = sin(x);
        r2[index] = cos(x);
    }
}

// S = 2*sqrt(r1^2+r2^2) = 2
__global__
void SexactD(Qreal* S, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        S[index] = 2.f;
    }
}

__global__
void wexactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = -2.f*sin(x+y);
    }
}

__global__
void uexactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = -cos(x+y);
    }
}
__global__
void vexactD(Qreal* phys, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = cos(x+y);
    }
}

__global__
void NL1exact(Qreal* phys, Qreal lambda, Qreal Pe, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = 0.5*( cos(y) + cos(2*x+y) +
        4.f*(lambda+cos(x))*sin(x+y) - 8.f*sin(x)*cn*cn/Pe );
    }
}

__global__
void NL2exact(Qreal* phys, Qreal lambda, Qreal Pe, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = -1.f*sin(x)*( cos(x+y) + 2.f*sin(x+y) ) - 4*cos(x)*cn*cn/Pe;
    }
}

__global__
void NL0exact(Qreal* phys, Qreal lambda, Qreal Pe, Qreal Er, Qreal Re, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    if (i<Nx && j<Ny){
        phys[index] = -1.f*cos(x)*(1+2*lambda+6*lambda*cn*cn)/(Er*Re);
    }
}

__global__
void Single1exact(Qreal* phys, Qreal lambda, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    // Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = sin(x)*(1+2.f*lambda + 6.f*lambda*cn*cn);
    }
}

__global__
void Single2exact(Qreal* phys, Qreal lambda, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = cos(x)*(1+2.f*lambda + 6.f*lambda*cn*cn);
    }
}

__global__
void crossexact(Qreal* phys, Qreal lambda, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = 0.f;
    }
}

inline void init(Field* phi, Field* wa, Field* ua, Field* va, Field* r1, Field* r2, 
Field* Sa, Field* single1a, Field* single2a, Field* NL0a, Field* NL1a, Field* NL2a, Field* crossa, 
Qreal lambda, Qreal cn, Qreal Pe, Qreal Er, Qreal Re){
    Mesh* mesh = r1->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ; 
    Qreal dx = mesh->dx; Qreal dy = mesh->dy;
    dim3 dimGrid = mesh->dimGridp; dim3 dimBlock = mesh->dimBlockp;

    PhiinitD<<<dimGrid, dimBlock>>>(phi->phys, Nx, Ny, BSZ, dx, dy);
    rinitD<<<dimGrid, dimBlock>>>(r1->phys, r2->phys, Nx, Ny, BSZ, dx, dy);
    SexactD<<<dimGrid, dimBlock>>>(Sa->phys, Nx, Ny, BSZ, dx, dy);
    wexactD<<<dimGrid, dimBlock>>>(wa->phys, Nx, Ny, BSZ, dx, dy);
    uexactD<<<dimGrid, dimBlock>>>(ua->phys, Nx, Ny, BSZ, dx, dy);
    vexactD<<<dimGrid, dimBlock>>>(va->phys, Nx, Ny, BSZ, dx, dy);
    Single1exact<<<dimGrid, dimBlock>>>(single1a->phys, lambda, cn, Nx, Ny, BSZ, dx, dy);
    Single2exact<<<dimGrid, dimBlock>>>(single2a->phys, lambda, cn, Nx, Ny, BSZ, dx, dy);
    NL0exact<<<dimGrid, dimBlock>>>(NL0a->phys, lambda, Pe, Er, Re, cn, Nx, Ny, BSZ, dx, dy);
    NL1exact<<<dimGrid, dimBlock>>>(NL1a->phys, lambda, Pe, cn, Nx, Ny, BSZ, dx, dy);
    NL2exact<<<dimGrid, dimBlock>>>(NL2a->phys, lambda, Pe, cn, Nx, Ny, BSZ, dx, dy);
    crossexact<<<dimGrid, dimBlock>>>(crossa->phys, lambda, cn, Nx, Ny, BSZ, dx, dy);
    // update the spectral
    FwdTrans(mesh, r1->phys, r1->spec);
    FwdTrans(mesh, r2->phys, r2->spec);

    // we want to test the currfunc so that we set the physics zero
    FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(r1->phys, 0.f, mesh->Nx, mesh->Ny, mesh->BSZ);
    FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(r2->phys, 0.f, mesh->Nx, mesh->Ny, mesh->BSZ);
    
    // update the spectral of vorticity
    FwdTrans(mesh, phi->phys, phi->spec);
}

__global__
void p11exact(Qreal* phys, Qreal lambda, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = sin(x)*(1+2*lambda+6*lambda*cn*cn);
    }
}

__global__
void p12exact(Qreal* phys, Qreal lambda, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = cos(x)*(1+2*lambda+6*lambda*cn*cn);
    }
}

__global__
void p21exact(Qreal* phys, Qreal lambda, Qreal cn, int Nx, int Ny, int BSZ, Qreal dx, Qreal dy){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nx;
    Qreal x = i*dx;
    Qreal y = j*dy;
    Qreal s = exp(-1*( (x-M_PI)*(x-M_PI)+(y-M_PI)*(y-M_PI) ));
    if (i<Nx && j<Ny){
        phys[index] = cos(x)*(1+2*lambda+6*lambda*cn*cn);
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

;void coord(Mesh &mesh){
    ofstream xcoord("x.csv");
    ofstream ycoord("y.csv");
    for (int j=0; j<mesh.Ny; j++){
        for ( int i=0; i< mesh.Nx; i++){
            Qreal x = mesh.dx*i;
            Qreal y = mesh.dy*j;
            xcoord << x << ",";
            ycoord << y << ",";
        }
        xcoord << endl;
        ycoord << endl;
    }
    xcoord.close();
    ycoord.close();
}

// we test the single func and the cross term func in
// this module, with the velocity computation. 
// the r1 is set to be 1/2*sin(x+y)*x/2pi and the r2 is set to be 1/2*cos(x+y)*y/2pi
// which satisfies that r1^2+r2^2<=0.25. S is derived to be 1/(16pi)*(x^2+y^2);
// lap(r1) = 1/(2*pi)*(cos(x+y) - x*sin(x+y)), 
// lap(r2) = 1/(2*pi)*(-1)*(sin(x+y) + y*cos(x+y))
// the cross term should be 2*(r2*Lap(r1) - r1*Lap(r2)) = 
int main(){
    int BSZ = 16;
    int Ns = 1000;
    int Nx = 512*3/2; // same as colin
    int Ny = Nx;
    int Nxh = Nx/2+1;
    Qreal Lx = 2*M_PI;
    Qreal Ly = 2*M_PI;
    Qreal dx = 2*M_PI/Nx;
    Qreal dy = 2*M_PI/Ny;
    Qreal lambda = 0.5;
    Qreal cn = 0.1;
    Qreal Pe = 0.3;
    Qreal Re = 5.f;
    Qreal Er = 6.f;
    Qreal dt = 0.05; // same as colin
    Qreal a = 1.0;

/////////// 2 ////////////
// we previously have verified the validity of laplacian and vel_func.
// in this file we test the func about the Q tensor (components, r1, r2) and 
// the intermediate components p (p11, p12, p21)
    Mesh *mesh = new Mesh(BSZ, Nx, Ny, Lx, Ly);
    Field* phi = new Field(mesh);
    Field* w = new Field(mesh); Field* wa = new Field(mesh);
    Field* u = new Field(mesh); Field* ua = new Field(mesh);
    Field* v = new Field(mesh); Field* va = new Field(mesh);

    Field *r1 = new Field(mesh); Field *r2 = new Field(mesh);
    Field *S = new Field(mesh); Field *Sa = new Field(mesh);
    
    Field *single1 = new Field(mesh); Field *single1a = new Field(mesh);
    Field *single2 = new Field(mesh); Field *single2a = new Field(mesh);
    Field *cross1 = new Field(mesh); Field *cross2 = new Field(mesh); 
    Field *crossa = new Field(mesh);

    Field *p11 = new Field(mesh); Field *p11a = new Field(mesh);
    Field *p12 = new Field(mesh); Field *p12a = new Field(mesh);
    Field *p21 = new Field(mesh); Field *p21a = new Field(mesh);

    Field *nl0 = new Field(mesh); Field *nl0a = new Field(mesh);
    Field *nl1 = new Field(mesh); Field *nl1a = new Field(mesh);
    Field *nl2 = new Field(mesh); Field *nl2a = new Field(mesh);
    Field *alpha = new Field(mesh);

    // aux is the abbreviation of auxiliary, where only act as intermediate values
    // to assist computation. So we should guarantee that it doesnt undertake any 
    // long term memory work.
    Field *aux = new Field(mesh); Field *aux1 = new Field(mesh);
    // Field* phi = new Field(mesh);
    // Field* w = new Field(mesh); Field* wa = new Field(mesh);
    // Field* u = new Field(mesh); Field* ua = new Field(mesh);
    // Field* v = new Field(mesh); Field* va = new Field(mesh);

    coord(*mesh);
    init(phi, wa, ua, va, r1, r2, Sa, single1a, single2a, nl0a, nl1a, nl2a, crossa, 
    lambda, cn, Pe, Er, Re);
    FldSet<<<mesh->dimGridp, mesh->dimBlockp>>>(alpha->phys, 1.f, Nx, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(phi, "phi.csv");
    field_visual(wa, "wa.csv");
    field_visual(ua, "ua.csv");
    field_visual(va, "va.csv");
    field_visual(alpha, "alpha.csv");

    field_visual(r1, "r10.csv");
    field_visual(r2, "r20.csv");
    field_visual(Sa, "Sa.csv");

    field_visual(single1a, "single1a.csv");
    field_visual(single2a, "single2a.csv");
    field_visual(nl0a, "nl0a.csv");
    field_visual(nl1a, "nl1a.csv");
    field_visual(nl2a, "nl2a.csv");
    
    // evaluate the spectral of w: Laplacian( Four(\phi) )
    laplacian_func(phi->spec, w->spec, mesh);
    // switch back
    BwdTrans(mesh, w->spec, w->phys);
    // evaluate the S; the velocity u, v; the phys of r1, r2; 
    curr_func(r1, r2, w, u, v, S);
    // evaluate the S
    // S_func(r1, r2, S);
    // S_funcD<<<mesh->dimGridp, mesh->dimBlockp>>>(r1->phys, r2->phys, S->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(r1, "r1.csv");
    field_visual(r2, "r2.csv");
    field_visual(w, "w.csv");
    field_visual(u, "u.csv");
    field_visual(v, "v.csv");
    field_visual(S, "S.csv");
//////////////////// Sfunc tested //////////////////

//////////////////// crossfunc test ///////////////
// Cross(r1,r2) = 2*(r2*\Delta(r1) - r1*\Delta(r2))
//  we here tested different functions where r1 = cos(2x+y), r2 = sin(x+3y) 
// for cross terms, where the exact value should be: 10cos(2x+y)sin(x+3y).

    pCross_func(cross1, aux1, r1, r2);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(cross1, "cross1.csv");

    pCross_func(cross2, aux1, r2, r1);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(cross2, "cross2.csv");

    pSingle_func(single1, aux, r1, S, alpha, lambda, cn);
    pSingle_func(single2, aux, r2, S, alpha, lambda, cn);
    pCross_func(cross1, aux, r1, r2);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(single1, "single1.csv");
    field_visual(single2, "single2.csv");
    field_visual(cross1, "cross1.csv");
//////////////////// cross and single term tested //////////////////

//////////////////// nonlinear term test //////////////////
    p11exact<<<mesh->dimGridp, mesh->dimBlockp>>>(p11a->phys, lambda, cn, Nx, Ny, BSZ, dx, dy);
    p12exact<<<mesh->dimGridp, mesh->dimBlockp>>>(p12a->phys, lambda, cn, Nx, Ny, BSZ, dx, dy);
    p21exact<<<mesh->dimGridp, mesh->dimBlockp>>>(p21a->phys, lambda, cn, Nx, Ny, BSZ, dx, dy);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(p11a, "p11a.csv");
    field_visual(p12a, "p12a.csv");
    field_visual(p21a, "p21a.csv");

    r1nonl_func(nl1, aux, r1, r2, w, u, v, S, lambda, cn, Pe);
    r2nonl_func(nl2, aux, r1, r2, w, u, v, S, lambda, cn, Pe);
    p11nonl_func(p11, aux, aux1, r1, r2, S, alpha, lambda, cn);
    p12nonl_func(p12, aux, aux1, r1, r2, S, alpha, lambda, cn);
    p21nonl_func(p21, aux, aux1, r1, r2, S, alpha, lambda, cn);
    cuda_error_func( cudaDeviceSynchronize() );

    wnonl_func(nl0, aux, aux1, p11, p12, p21, r1, r2, 
    w, u, v, alpha, S, Re, Er, cn, lambda);

    FwdTrans(mesh, p12->phys, p12->spec);
    xDeriv(p12->spec, aux->spec, mesh);
    cuda_error_func( cudaDeviceSynchronize() );
    xDeriv(aux->spec, aux->spec, mesh);
    dealiasing_func<<<mesh->dimGridsp, mesh->dimBlocksp>>>(aux->spec, mesh->cutoff, Nxh, Ny, BSZ);
    BwdTrans(mesh, aux->spec, aux->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(aux, "Dxxp12.csv");   
    BwdTrans(nl0->mesh, nl0->spec, nl0->phys);
    cuda_error_func( cudaDeviceSynchronize() );
    field_visual(nl1, "nl1.csv");
    field_visual(nl2, "nl2.csv");
    field_visual(p11, "p11.csv");
    field_visual(p12, "p12.csv");
    field_visual(p21, "p21.csv");
    field_visual(nl0, "nl0.csv");

    return 0;
}