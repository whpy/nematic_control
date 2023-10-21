#include <TimeIntegration/RK4.cuh>



//RK4 integrating steps
// du/dt = \alpha*u + F(t,u)
// IFh = exp(\alpha*dt/2). IF = exp(\alpha*dt)
// u_{n+1} = u_{n}*IF + 1/6*(a*IF + 2b*IFh + 2c*IFh + d)
// a_n = dt*F(t_n,u_n)
// b_n = dt*F(t_n+dt/2, (u_n+a_n/2)*IFh)
// c_n = dt*F(t_n+dt/2, u_n*IFh + b_n/2)
// d_n = dt*F(t_n+dt, u_n*IF + c_n*IFh)
// 
// Here, u_old represents u_n, u_curr represents the input for computing
// the intermediate variables (a,b,c,d), u_new represents the value at next
// step. 

// preparation before RK4 integration
// compute the value at next step adding the first term u_{n}*IF of the summation 
// prepare input ucurr(u_n) for computation of a_n
void integrate_func0(Field *u_old, Field *u_curr, Field *u_new, float dt, 
float *IF, float *IFh){
    Mesh* mesh = u_old->mesh;
    int Nxh = mesh->Nxh;
    int Ny = mesh->Ny;
    int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridsp;
    dim3 dimBlock = mesh->dimBlocksp; 
    // u_{n+1} = u_{n}*exp(alpha * dt)
    SpecMul<<<dimGrid, dimBlock>>>(u_old->spec, IF, 1., u_new->spec, Nxh, Ny, BSZ);
    // u_curr = u_{n}
    SpecSet<<<dimGrid, dimBlock>>>(u_curr->spec, u_old->spec, Nxh, Ny, BSZ);
}
// __global_
// void integrate_func0(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new,
//                     float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt){
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     if(i < Nxh && j < Ny){
//         // u_{n+1} = u_{n}*exp(alpha * dt)
//         spec_new[index] = spec_old[index]*IF[index];
//         // u_{n}
//         spec_curr[index] = spec_old[index];
//     }
// }
void integrate_func1(Field *u_old, Field *u_curr, Field *u_new, Field* u_nonl, float dt, 
float *IF, float *IFh){
    Mesh* mesh = u_old->mesh;
    int Nxh = mesh->Nxh;
    int Ny = mesh->Ny;
    int BSZ = mesh->BSZ;
    dim3 dimGrid = mesh->dimGridsp;
    dim3 dimBlock = mesh->dimBlocksp; 
    integrate_func1D<<<dimGrid, dimBlock>>>(u_old->spec, u_curr->spec, u_new->spec, u_nonl->spec, 
    IF, IFh, Nxh, Ny, BSZ, dt);
}
__global__ 
void integrate_func1D(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
            float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    
    cuComplex an = spec_nonl[index]*dt;
    
    // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n)
    spec_new[index] = spec_new[index] + float(1/6*IF[index]) * an;
    printf("(%f,%f),%f \n", an.x, an.y, float(1/6)*IF[index]);
    // u_curr = (u_{n}+a_{n}/2)*exp(alpha*dt/2)
    spec_curr[index] = (spec_old[index]+an/2) * IFh[index];
}


// __global__ 
// void integrate_func1(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, cuComplex* spec_nonl,
//                     float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt){
//     // spec_nonl = a_n/dt here
//     // spec_curr represents the value to be input into Nonlinear function for b_n/dt next 
//     int i = blockIdx.x * BSZ + threadIdx.x;
//     int j = blockIdx.y * BSZ + threadIdx.y;
//     int index = j*Nxh + i;
//     if(i < Nxh && j < Ny){
//         cuComplex an = spec_nonl[index]*dt;
//         // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n)
//         spec_new[index] = spec_new[index] + 1/6*IF[index] * an;
//         // (u_{n}+a_{n}/2)*exp(alpha*dt/2)
//         spec_curr[index] = (spec_old[index]+an/2) * IFh[index];
//     }
// }
__global__ void integrate_func2(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt){
    // spec_nonl = b_n/dt here
    // spec_curr represents the value to be input into Nonlinear function for c_n/dt next 
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        cuComplex bn = spec_nonl[index]*dt;
        // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n) + 1/6*exp(alpha*dt/2)*(b_n)
        spec_new[index] = spec_new[index] + 1/3*IFh[index] * bn;
        // (u_{n}*exp(alpha*dt/2) + b_{n}/2)
        spec_curr[index] = (spec_old[index]*IFh[index] + bn/2) ;
    }
}
__global__ void integrate_func3(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt){
    // spec_nonl = c_n/dt here
    // spec_curr represents the value to be input into Nonlinear function for d_n/dt next 
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        cuComplex cn = spec_nonl[index]*dt;
        // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n) + 1/3*exp(alpha*dt/2)*(b_n) 
        //         + 1/3*exp(alpha*dt/2)*(c_n)
        spec_new[index] = spec_new[index] + 1/3*IFh[index] * cn;
        // u_{n}*exp(alpha*dt) + c_{n} * exp(alpha*dt/2)
        spec_curr[index] = (spec_old[index]*IF[index] + cn*IFh[index]) ;
    }
}
__global__ void integrate_func4(cuComplex* spec_old, cuComplex* spec_curr, cuComplex* spec_new, 
                        cuComplex* spec_nonl,float* IF, float* IFh, int Nxh, int Ny, int BSZ, float dt){
    // spec_nonl = d_n/dt here
    // spec_curr represents the value to be input into Nonlinear function for d_n/dt next 
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i < Nxh && j < Ny){
        cuComplex dn = spec_nonl[index]*dt;
        // u_{n+1} = u_{n}*exp(alpha * dt) + 1/6*exp(alpha*dt)*(a_n) + 1/3*exp(alpha*dt/2)*(b_n) 
        //         + 1/3*exp(alpha*dt/2)*(c_n) + 1/6*d_n
        spec_new[index] = spec_new[index] + 1/6*dn;
    }

}