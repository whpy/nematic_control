#include <Basic/FldOp.cuh>


/******************************************************************************* 
 * operator functions                                                          * 
 *******************************************************************************/
// operators for field solvers
// divide a factor after transforming from spectral to physical
__global__ void coeff(float *f, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]/(Nx*Ny);
    }
}

//update the spectral space based on the value in physical space
void FwdTrans(Mesh* pmesh, float* phys, cuComplex* spec){
    cufft_error_func( cufftExecR2C(pmesh->transf, phys, spec));
    cuda_error_func( cudaDeviceSynchronize() );
}

//update the physics space based on the value in spectral space
void BwdTrans(Mesh* pmesh, cuComplex* spec, float* phys){
    int Nx = pmesh->Nx;
    int Ny = pmesh->Ny;
    int BSZ = pmesh->BSZ;
    cufft_error_func( cufftExecC2R(pmesh->inv_transf, spec, phys));
    cuda_error_func( cudaDeviceSynchronize() );
    dim3 dimGrid = pmesh->dimGridp;
    dim3 dimBlock = pmesh->dimBlockp;
    coeff<<<dimGrid, dimBlock>>>(phys, Nx, Ny, BSZ);
    // in the referenced source code, they seem a little bit abuse synchronization, this
    // may be a point that we could boost the performance in the future. we temporarily
    // comply with the same principle that our code would at least perform no worse than theirs
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__
// __Device: phys field multiplication: pc(x) = C*pa(x,y)*pb(x,y). prefix p denotes physical
void FldMul(float* pa, float* pb, float C, float* pc, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = C*pa[index]*pb[index];
    }
}

__global__ 
// __Device: phys field addition: pc(x,y) = a*pa(x,y) + b*pb(x,y). prefix p denotes physical
void FldAdd(float* pa, float* pb, float a, float b, float* pc, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b*pb[index];
    }
}

__global__
void FldAdd(float a, float* pa, float b, float* pb, float* pc, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b*pb[index];
    }
}
__global__
// set physical field equals to a constant field
void FldSet(float * pf, float c, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if (i<Nx && j<Ny){
        pf[index] = c;
    }
}

// create a constant field pf = c
__global__
void FldSet(float * pf, float* c, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if (i<Nx && j<Ny){
        pf[index] = c[index];
    }
}

// set two physical field equals, pa = pb
__global__
void SpecSet(cuComplex * pa, cuComplex* pb, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        pa[index] = pb[index];
    }
}

__global__
void SpecSet(cuComplex * pa, cuComplex c, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        pa[index] = c;
    }
}

__global__ 
// spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
void SpecAdd(cuComplex* spa, cuComplex* spb, float a, float b, cuComplex* spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = a* spa[index] + b* spb[index];
    }
}

__global__
void SpecAdd(float a, cuComplex* spa, float b, cuComplex* spb, cuComplex* spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = a* spa[index] + b* spb[index];
    }
}

__global__ void xDerivD(cuComplex *ft, cuComplex *dft, float* kx, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*kx[i];
    }
}
void xDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh){
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    xDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->kx, mesh->Nxh, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__ void yDerivD(cuComplex *ft, cuComplex *dft, float* ky, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*ky[j];
    }
}
void yDeriv(cuComplex *ft, cuComplex *dft, Mesh *mesh){
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    yDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->ky,mesh->Nxh, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}
