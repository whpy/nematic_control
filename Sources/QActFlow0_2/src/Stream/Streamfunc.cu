#include <Stream/Streamfunc.cuh>

__global__ void vel_funcD(cuComplex* w_spec, cuComplex* u_spec, cuComplex* v_spec, 
                            float* k_squared, float* kx, float*ky, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i==0 && j==0)
    {
        u_spec[index] = make_cuComplex(0.,0.);
        v_spec[index] = make_cuComplex(0.,0.);
    }
    else if(i<Nxh && j<Ny){
        //u = -D_y(\phi) -> u_spec = -1 * i* ky* w_spec/(-1* (kx^2+ky^2) )
        u_spec[index] = -1.f * ky[j]*im()*w_spec[index]/(-1.f*k_squared[index]);
        //v = D_x(\phi) -> v_spec = -1 * i* kx* w_spec/(-1* (kx^2+ky^2) )
        v_spec[index] = kx[i]*im()*w_spec[index]/(-1.f*k_squared[index]);
    }
}
void vel_func(Field* w, Field* u, Field* v){
    int Nxh = w->mesh->Nxh;
    int Ny = w->mesh->Ny;
    int BSZ = w->mesh->BSZ;
    float* k_squared = w->mesh->k_squared;
    float* kx = w->mesh->kx;
    float* ky = w->mesh->ky;
    dim3 dimGrid = w->mesh->dimGridsp;
    dim3 dimBlock = w->mesh->dimBlocksp; 
    vel_funcD<<<dimGrid, dimBlock>>>(w->spec, u->spec, v->spec, k_squared, kx, ky, Nxh, Ny, BSZ);
    BwdTrans(u->mesh, u->spec, u->phys);
    BwdTrans(v->mesh, v->spec, v->phys);
}

__global__ 
void S_funcD(float* r1, float* r2, float* S, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        S[index] = 2*sqrt(r1[index]*r1[index] + r2[index]*r2[index]);
    }
}
void S_func(Field* r1, Field* r2, Field* S){
    Mesh* mesh = S->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridp; dim3 dimBlock = mesh->dimBlockp;
    S_funcD<<<dimGrid, dimBlock>>>(r1->phys, r2->phys, S->phys, Nx, Ny, BSZ);
}

void curr_func(Field *r1curr, Field *r2curr, Field *wcurr, Field *u, Field *v, Field *S){
    // obtain the physical values of velocities and r_i
    int Nx = r1curr->mesh->Nx; int Ny = r1curr->mesh->Ny; int BSZ = r1curr->mesh->BSZ;
    dim3 dimGrid = r1curr->mesh->dimGridp; dim3 dimBlock = r1curr->mesh->dimBlockp;

    vel_func(wcurr, u, v);
    BwdTrans(r1curr->mesh,r1curr->spec, r1curr->phys);
    BwdTrans(r2curr->mesh,r2curr->spec, r2curr->phys);
    BwdTrans(wcurr->mesh, wcurr->spec, wcurr->phys);
    // calculate the physical val of S
    S_funcD<<<dimGrid, dimBlock>>>(r1curr->phys, r2curr->phys, S->phys, Nx, Ny, BSZ);
}

__global__
void r1lin_func(float* IFr1h, float* IFr1, float* k_squared, float Pe, float cn, float dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        IFr1h[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt/2);
        IFr1[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt);
    }
}

__global__
void r2lin_func(float* IFr2h, float* IFr2, float* k_squared, float Pe, float cn, float dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        IFr2h[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt/2);
        IFr2[index] = exp((1.f/Pe*(-1.f*k_squared[index]+cn*cn))*dt);
    }
}

__global__
void wlin_func(float* IFwh, float* IFw, float* k_squared, float Re, float cf, float dt, int Nxh, int Ny, int BSZ)
{
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        IFwh[index] = exp( ((-1.0*k_squared[index])-1.0) *dt/2);
        IFw[index] = exp( ((-1.0*k_squared[index])-1.0) *dt);
    }
}

// calculate the cross term in pij where Cross(r1,r2) = 2*(r2*\Delta(r1) - r1*\Delta(r2))
void pCross_func(Field *p, Field *aux, Field *r1, Field *r2){
    // this function only works on the physical space
    int Nx = p->mesh->Nx;
    int Ny = p->mesh->Ny;
    int Nxh = p->mesh->Nxh;
    int BSZ = p->mesh->BSZ;
    Mesh* mesh = p->mesh;
    dim3 dimGrid = p->mesh->dimGridp;
    dim3 dimBlock = p->mesh->dimBlockp;
    //aux->spec = Four(/Delta(r1))
    laplacian_func(r1->spec,aux->spec,aux->mesh);
    //aux->phys = /Delta(r1)
    BwdTrans(aux->mesh, aux->spec, aux->phys);
    
    // p->phys = 2* r2*\Delta(r1)
    FldMul<<<dimGrid, dimBlock>>>(aux->phys,r2->phys, 2.f, p->phys, Nx, Ny, BSZ);

    //aux.spec = Four(/Delta(r2))
    laplacian_func(r2->spec,aux->spec,aux->mesh);
    //aux->phys = /Delta(r2)
    BwdTrans(aux->mesh, aux->spec, aux->phys);
    // aux->phys = -2* r1*\Delta(r2)
    FldMul<<<dimGrid, dimBlock>>>(aux->phys,r1->phys, -2.f, aux->phys, Nx, Ny, BSZ);

    // p.phys = aux.phys + p.phys = -2* r2*\Delta(r1) + 2* r2*\Delta(r1)
    FldAdd<<<dimGrid, dimBlock>>>(1.f, p->phys, 1.f, aux->phys, p->phys, Nx, Ny, BSZ);
    // cross term physical value update successfully
}

// the rest term of the pij where 
//Single(ri) = \alpha*ri + \lambda* S*(cn^2*(S^2-1)*ri - \Delta ri) 
//           = \alpha*ri - \lambda* S* \Delta(ri) 
//            + \lambda* S*cn^2*(S^2)*ri - \lambda* S*cn^2*ri
void pSingle_func(Field *p, Field *aux, Field *r, Field *S, Field *alpha, float lambda, float cn){
    // this function only works on the physical space
    int Nx = p->mesh->Nx;
    int Ny = p->mesh->Ny;
    int BSZ = p->mesh->BSZ;
    dim3 dimGrid = p->mesh->dimGridp;
    dim3 dimBlock = p->mesh->dimBlockp;
    
    //aux.phys = \alpha* ri
    FldMul<<<dimGrid, dimBlock>>>(r->phys, alpha->phys, 1.f, aux->phys, Nx, Ny, BSZ);
    
    // -\lambda* S* \Delta(ri)
    // p.spec = \Delta(r)
    laplacian_func(r->spec, p->spec, p->mesh);
    // p.phys = \Delta(r)
    BwdTrans(p->mesh, p->spec, p->phys);

    // p.phys = -\lambda*\Delta(r)*S
    FldMul<<<dimGrid, dimBlock>>>(p->phys, S->phys, -1.f*lambda, p->phys, Nx, Ny, BSZ);
    // p.phys = p.phys + aux.phys = \alpha* ri -\lambda*\Delta(r)*S
    FldAdd<<<dimGrid, dimBlock>>>(1.f, aux->phys, 1.f, p->phys, p->phys, Nx, Ny, BSZ);

    // \lambda*cn^2*(S^3)*ri
    // aux.phys = \lambda* cn^2 * S * S
    FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, lambda*cn*cn, aux->phys, Nx, Ny, BSZ);
    // aux.phys =aux.phys* S = \lambda* cn^2 * S * S* S
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, S->phys, 1.f, aux->phys, Nx, Ny, BSZ);
    // aux.phys =aux.phys* ri = \lambda* cn^2 * S * S* S* ri
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, r->phys, 1., aux->phys, Nx, Ny, BSZ);
    // p.phys = p.phys + aux.phys = \alpha* ri -\lambda*\Delta(r)*alpha + \lambda*cn^2*S^3*ri
    FldAdd<<<dimGrid, dimBlock>>>(1.f, aux->phys, 1.f, p->phys, p->phys, Nx, Ny, BSZ);

    // -\lambda* S*cn^2*ri
    // aux.phys = -1*\lambda* cn^2 * S * ri
    FldMul<<<dimGrid, dimBlock>>>(S->phys, r->phys, -1*lambda*cn*cn, aux->phys, Nx, Ny, BSZ);
    // p.phys = p.phys + aux.phys 
    // = \alpha* ri -\lambda*\Delta(r)*alpha + \lambda*cn^2*S^3*ri + (-1*\lambda*cn^2 *S*ri)
    FldAdd<<<dimGrid, dimBlock>>>(1.f, aux->phys, 1.f, p->phys, p->phys, Nx, Ny, BSZ);
    // cross term physical value update successfully
}

// p11 = Single(r1)
void p11nonl_func(Field *p11, Field* aux, Field* aux1, Field* r1, Field *r2, Field *S, 
                        Field *alpha, float lambda, float cn){
    // our strategy is firstly update the phys then finally update the spectral
    // p11.phys = \lambda* S(cn^2*(S^2-1)*r1 - \Delta r1) + \alpha*r1
    pSingle_func(p11, aux, r1, S, alpha, lambda,cn);
    cuda_error_func( cudaDeviceSynchronize() );
    FwdTrans(p11->mesh, p11->phys, p11->spec);
    // p11 spectral update finished
}
// p12 = Cross(r1,r2) + Single(r2)
void p12nonl_func(Field *p12, Field *aux, Field *aux1, Field *r1, Field *r2, Field *S, 
                        Field *alpha, float lambda, float cn){
    int Nx = p12->mesh->Nx;
    int Ny = p12->mesh->Ny;
    int BSZ = p12->mesh->BSZ;
    dim3 dimGrid = p12->mesh->dimGridp;
    dim3 dimBlock = p12->mesh->dimBlockp;
    // p12.phys = Cross(r1,r2) = 2*(r2*\Delta(r1) - r1*\Delta(r2))
    pCross_func(p12, aux, r1, r2);
    // aux.phys = Single(r2) = \lambda* S(cn^2*(S^2-1)*r2 - \Delta r2) + \alpha*r2
    pSingle_func(aux, aux1, r2, S, alpha, lambda,cn);
    // p12.phys = p12.phys + aux.phys = Cross(r1,r2) + Single(r2)
    FldAdd<<<dimGrid, dimBlock>>>(1.f, p12->phys, 1.f, aux->phys, p12->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    FwdTrans(p12->mesh, p12->phys, p12->spec);
    cuda_error_func( cudaDeviceSynchronize() );
    // p12 spectral update finished
}

// p22 = -1*Cross(r1,r2) + Single(r2)
void p21nonl_func(Field *p21, Field *aux, Field *aux1, Field *r1, Field *r2, Field *S, 
                        Field *alpha, float lambda, float cn){
    int Nx = p21->mesh->Nx;
    int Ny = p21->mesh->Ny;
    int BSZ = p21->mesh->BSZ;
    dim3 dimGrid = p21->mesh->dimGridp;
    dim3 dimBlock = p21->mesh->dimBlockp;
    // p21.phys = Cross(r2,r1) = 2*(r1*\Delta(r2) - r2*\Delta(r1))
    pCross_func(p21, aux, r2, r1);
    // aux.phys = Single(r2) = \lambda* S(cn^2*(S^2-1)*r2 - \Delta r2) + \alpha*r2
    pSingle_func(aux, aux1, r2, S, alpha, lambda, cn);
    // p21.phys = p21.phys + aux.phys = Cross(r2,r1) + Single(r2)
    FldAdd<<<dimGrid, dimBlock>>>(1.f, p21->phys, 1.f, aux->phys, p21->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaDeviceSynchronize() );
    FwdTrans(p21->mesh, p21->phys, p21->spec);
    cuda_error_func( cudaDeviceSynchronize() );
    // p21 spectral update finished
}


void r1nonl_func(Field *r1nonl, Field *aux, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *S, float lambda, float cn, float Pe){
    // non-linear for r1: 
    // \lambda S\frac{\partial u}{\partial x}  + (-1* \omega_z* r2) + (-cn^2/Pe *S^2*r1)
    // + (-1* u* D_x\omega_z) + (-1*v*D_y(\omega_z))
    Mesh *mesh = r1nonl->mesh;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;

    // \lambda S\frac{\partial u}{\partial x}
    //aux.spec = \partial_x u
    xDeriv(u->spec, aux->spec, r1nonl->mesh);
    //aux.phys = \partial_x u
    BwdTrans(r1nonl->mesh, aux->spec, aux->phys);
    // r1nonl.phys = \lambda*S(x,y) * aux = \lambda*S(x,y) * \partial_x u(x,y) 
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, S->phys, lambda, r1nonl->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    // (-\omega_z* r2)
    // aux.phys = -1*\omega*r2
    FldMul<<<dimGrid, dimBlock>>>(w->phys, r2->phys, -1.0, aux->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );
    // r1nonl.phys = \lambda*S(x,y) * \partial_x u(x,y) + (-\omega_z* r2)
    FldAdd<<<dimGrid, dimBlock>>>(r1nonl->phys, aux->phys, 1.f, 1.f, r1nonl->phys, Nx, Ny, BSZ);

    //(-cn^2/Pe *S^2*r1)
    // aux.phys = -1*cn^2/Pe*S*S
    FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.f*cn*cn/Pe, aux->phys, Nx, Ny, BSZ);
    // aux.phys = -1*cn^2/Pe*S*S*r1
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, r1->phys, 1.f, aux->phys, Nx, Ny, BSZ);
    // r1nonl.phys = \lambda S\frac{\partial u}{\partial x}  + (-\omega_z* r2) + (-1*cn^2/Pe*S*S*r1)
    FldAdd<<<dimGrid, dimBlock>>>(r1nonl->phys, aux->phys, 1., 1., r1nonl->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    //(-u*D_x(r1))
    // aux.spec = i*kx*r1
    xDeriv(r1->spec, aux->spec, w->mesh);
    // aux.phys = D_x(r1)
    BwdTrans(aux->mesh,aux->spec, aux->phys);
    // aux.phys = -1*aux.phys*u(x,y) = -1*D_x(r1)*u(x,y)
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, u->phys, -1.f, aux->phys, Nx, Ny, BSZ);
    // r1nonl.phys =
    // \lambda S\frac{\partial u}{\partial x}  + (-\omega_z* r2) + (-1*cn^2/Pe*S*S*r1) 
    // + (-1*D_x(r1)*u(x,y))
    FldAdd<<<dimGrid, dimBlock>>>(r1nonl->phys, aux->phys, 1.0, 1.0, r1nonl->phys, Nx, Ny, BSZ);

    //(-1*v*D_y(\omega_z))
    // aux.spec = i*ky*r1
    yDeriv(r1->spec, aux->spec, w->mesh);
    // aux.phys = D_y(r1)
    BwdTrans(aux->mesh,aux->spec, aux->phys);
    // aux.phys = -1*aux.phys*v(x,y) = -1*D_y(r1)*v(x,y)
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, v->phys, -1.f, aux->phys, Nx, Ny, BSZ);
    // r1nonl.phys = r1nonl.phys + aux.phys = 
    // \lambda S\frac{\partial u}{\partial x}  + (-\omega_z* r2) + (-1*cn^2/Pe*S*S*r1) 
    // + (-1*D_x(w)*u(x,y)) + (-1*v*D_y(r1))
    FldAdd<<<dimGrid, dimBlock>>>(r1nonl->phys, aux->phys, 1.0, 1.0, r1nonl->phys, Nx, Ny, BSZ);

    // the spectral of r1 nonlinear term is calculated here based on the physical value
    // that evaluated before.
    FwdTrans(r1nonl->mesh, r1nonl->phys, r1nonl->spec);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize());
}

void r2nonl_func(Field *r2nonl, Field *aux, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *S, float lambda, float cn, float Pe){
    // non-linear for r2: 
    // \lambda* S* 1/2* (D_x(v)+D_y(u)) + (\omega_z* r1) + (-cn^2/Pe *S^2*r2)
    // + (-1* u* D_x(r2))) + (-1*v*D_y(r2))
    int Nx = r2nonl->mesh->Nx;
    int Ny = r2nonl->mesh->Ny;
    int BSZ = r2nonl->mesh->BSZ;
    dim3 dimGrid = r2nonl->mesh->dimGridp; dim3 dimBlock = r2nonl->mesh->dimBlockp;

    // \lambda* S* 1/2* (D_x(v))
    //aux.spec = \partial_x u
    xDeriv(v->spec, aux->spec, r2nonl->mesh);
    //aux.phys = \partial_x u
    BwdTrans(aux->mesh, aux->spec, aux->phys);
    // r2nonl.phys = \lambda*S(x,y) * aux = \lambda/2 *S(x,y) * D_x(u(x,y)) 
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, S->phys, lambda*0.5, r2nonl->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    // \lambda* S* 1/2 *(D_y(u))
    //aux.spec = D_y(u)
    yDeriv(u->spec, aux->spec, aux->mesh);
    //aux.phys = D_y(u)
    BwdTrans(aux->mesh, aux->spec, aux->phys);
    // aux.phys = \lambda* 1/2 *S(x,y) * aux = \lambda/2 *S(x,y) * D_y(u(x,y)) 
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, S->phys, lambda*0.5, aux->phys, Nx, Ny, BSZ);
    // r2nonl.phys = r2nonl.phys + aux.phys = 1/2*\lambda *S(x,y) *D_x(v(x,y)) + 1/2*\lambda *S(x,y) * D_y(u(x,y)) 
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl->phys, aux->phys, 1., 1., r2nonl->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    // (\omega_z* r1)
    // aux.phys = 1.0* \omega*r1
    FldMul<<<dimGrid, dimBlock>>>(w->phys, r1->phys, 1.0, aux->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );
    // r2nonl.phys = (\lambda/2 *S(x,y) *D_x(u(x,y))) + (\lambda/2 *S(x,y) * D_x(u(x,y))) + (\omega_z* r1)
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl->phys, aux->phys, 1.f, 1.f, r2nonl->phys, Nx, Ny, BSZ);

    //(-cn^2/Pe *S^2*r2)
    // aux.phys = -1*cn^2/Pe*S*S
    FldMul<<<dimGrid, dimBlock>>>(S->phys, S->phys, -1.f*cn*cn/Pe, aux->phys, Nx, Ny, BSZ);
    // aux.phys = -1*cn^2/Pe*S*S*r2
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, r2->phys, 1.f, aux->phys, Nx, Ny, BSZ);
    // r1nonl.phys = (\lambda/2 *S(x,y) *D_x(u(x,y))) + (\lambda/2 *S(x,y) * D_x(u(x,y))) + (\omega_z* r2) + (-1*cn^2/Pe*S*S*r2)
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl->phys, aux->phys, 1.f, 1.f, r2nonl->phys, Nx, Ny, BSZ);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize() );

    //(-u*D_x(r2))
    // aux.spec = i*kx*r2
    xDeriv(r2->spec, aux->spec, r2->mesh);
    // aux.phys = D_x(r2)
    BwdTrans(aux->mesh,aux->spec, aux->phys);
    // aux.phys = -1*aux.phys*u(x,y) = -1*D_x(r2)*u(x,y)
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, u->phys, -1.f, aux->phys, Nx, Ny, BSZ);
    // r2nonl.phys =
    // (\lambda/2 *S(x,y) *D_x(u(x,y))) + (\lambda/2 *S(x,y) * D_x(u(x,y))) + (\omega_z* r2) 
    //+ (-1*cn^2/Pe*S*S*r2) + (-1*D_x(w)*u(x,y))
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl->phys, aux->phys, 1.0, 1.0, r2nonl->phys, Nx, Ny, BSZ);

    //(-1*v*D_y(r2))
    // aux.spec = i*ky*r2
    yDeriv(r2->spec, aux->spec, r2->mesh);
    // aux.phys = D_y(r2)
    BwdTrans(aux->mesh,aux->spec, aux->phys);
    // aux.phys = -1*aux.phys*v(x,y) = -1*D_y(r2)*v(x,y)
    FldMul<<<dimGrid, dimBlock>>>(aux->phys, v->phys, -1.f, aux->phys, Nx, Ny, BSZ);
    // r2nonl.phys = r2nonl.phys + aux.phys = 
    // (\lambda/2 *S(x,y) *D_x(u(x,y))) + (\lambda/2 *S(x,y) * D_x(u(x,y))) + (\omega_z* r2) 
    // + (-1*cn^2/Pe*S*S*r2) + (-1*D_x(w)*u(x,y)) + (-1*v*D_y(\omega_z))
    FldAdd<<<dimGrid, dimBlock>>>(r2nonl->phys, aux->phys, 1.0, 1.0, r2nonl->phys, Nx, Ny, BSZ);

    // the spectral of r1 nonlinear term is calculated here based on the physical value
    // that evaluated before.
    FwdTrans(r2nonl->mesh, r2nonl->phys, r2nonl->spec);
    cuda_error_func( cudaPeekAtLastError() );
    cuda_error_func( cudaDeviceSynchronize());
}


void wnonl_func(Field *wnonl, Field *aux, Field *aux1, Field *p11, Field *p12, Field *p21, Field *r1, Field *r2, Field *w, 
                        Field *u, Field *v, Field *alpha, Field *S, float Re, float Er, float cn, float lambda){
            // wnonl = 1/ReEr * (D^2_xx(p12) - 2*D^2_xy(p11) - D^2_yy(p21))  
            //         + (-1* u*D_x(w)) + (-1* v* D_y(w)) 
    Mesh* mesh = wnonl->mesh;
    int BSZ = mesh->BSZ;
    int Nx = mesh->Nx; int Ny = mesh->Ny; int Nxh = mesh->Nxh;

    p11nonl_func(p11, aux, aux1, r1, r2, S, alpha, lambda, cn); 
    p12nonl_func(p12, aux, aux1, r1, r2, S, alpha, lambda, cn); 
    p21nonl_func(p21, aux, aux1, r1, r2, S, alpha, lambda, cn); 
    cuda_error_func( cudaDeviceSynchronize() );
    // aux.spec = D_x(p12)
    xDeriv(p12->spec, aux->spec, p12->mesh);
    // wnonl.spec = D^2_xx(p12)
    xDeriv(aux->spec, wnonl->spec, aux->mesh);
    
    // aux.spec = D_x(p11)
    xDeriv(p11->spec, aux->spec, p11->mesh);
    // aux.spec = D_y(aux.spec) = D^2_xy(p11)
    yDeriv(aux->spec, aux->spec, aux->mesh);
    // wnonl.spec = D^2_xx(p12) - 2*aux.spec = D^2_xx(p12) - 2*D^2_xy(p11)
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.f, wnonl->spec, -2.f, aux->spec, 
    wnonl->spec, wnonl->mesh->Nxh, wnonl->mesh->Ny, BSZ);

    // aux.spec = D_y(p21)
    yDeriv(p21->spec, aux->spec, p21->mesh);
    // aux.spec = D^2_yy(p12)
    yDeriv(aux->spec, aux->spec, aux->mesh);
    // wnonl.spec = D^2_xx(p12) - 2*D^2_xy(p11) - aux.spec 
    // = D^2_xx(p12) - 2*D^2_xy(p11) - D^2_yy(p21)
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.f, wnonl->spec, -1.f, aux->spec, 
    wnonl->spec, wnonl->mesh->Nxh, wnonl->mesh->Ny, BSZ);

    // aux.spec = D_x(w)
    xDeriv(w->spec, aux->spec, aux->mesh);
    // aux.phys = D_x(w)
    BwdTrans(aux->mesh, aux->spec, aux->phys);
    // aux.phys = (-1* u* D_x(w))
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(aux->phys, u->phys, -1.f, aux->phys, Nx, Ny, BSZ);
    // forward to the spectral: aux.spec = Four((-1* u* D_x(w)))
    FwdTrans(aux->mesh, aux->phys, aux->spec);
    // wnonl.spec = D^2_xx(p12) - 2*D^2_xy(p11) + aux.spec 
    // = D^2_xx(p12) - 2*D^2_xy(p11) - D^2_yy(p21) + (-1* u* D_x(w))
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1.f, wnonl->spec, 1.f, aux->spec, 
    wnonl->spec, wnonl->mesh->Nxh, wnonl->mesh->Ny,BSZ);

    // aux.spec = D_y(w)
    yDeriv(w->spec, aux->spec, aux->mesh);
    // aux.phys = D_y(w)
    BwdTrans(aux->mesh, aux->spec, aux->phys);
    // aux.phys = (-1* v* D_y(w))
    FldMul<<<mesh->dimGridp, mesh->dimBlockp>>>(aux->phys, v->phys, -1., aux->phys, Nx, Ny, BSZ);
    // forward to the spectral: aux.spec = Four((-1* v* D_y(w)))
    FwdTrans(aux->mesh, aux->phys, aux->spec);
    // wnonl.spec = D^2_xx(p12) - 2*D^2_xy(p11) + aux.spec 
    // = D^2_xx(p12) - 2*D^2_xy(p11) - D^2_yy(p21) + (-1* u* D_x(w)) + (-1* v* D_y(w))
    SpecAdd<<<mesh->dimGridsp, mesh->dimBlocksp>>>(1., wnonl->spec, 1., aux->spec, 
    wnonl->spec, wnonl->mesh->Nxh, wnonl->mesh->Ny, BSZ);

    cuda_error_func( cudaDeviceSynchronize() );
    // here the wnonl has updated sucessfully
    
}