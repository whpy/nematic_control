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
