#include <Basic/FldOp.cuh>


/******************************************************************************* 
 * operator functions                                                          * 
 *******************************************************************************/
// operators for field solvers
// divide a factor after transforming from spectral to physical
__global__
// generate the 0-1 sequence to consider which frequency to be deprecated
void cutoff_func(float* cutoff, int Nxh, int Ny, int BSZ);
// deprecate the high frequencies determined by the cutoff array
__global__
void dealiasing_func(cuComplex* f_spec, float* cutoff,int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        f_spec[index] = cutoff[index]*f_spec[index];
    }
}

__global__ void coeff(float *f, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]/(Nx*Ny);
    }
}

// in the referenced code, this function occurs more frequently than the dealiasing,
// it is applied after each time the nonlinear function is called. so maybe it is the
// main reason to retain the numerical precision.
__global__
void symmetry_func(cuComplex f_spec[], int Nxh, int Ny, int BSZ)
{
	int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
	
	// We make sure that the transformed array keeps the necessary symmetry for
	// RFT. (The 1st column should consist of pairs of complex conjugates.)
	// the first column due to the zero value of wavenumver in x, the sequence
	// degenerates to the trivial one-dimensional dft which need to fulfil symmetry
	// here size equal to (Nx/2+1)*Ny where the size of wave numbers.
	cuComplex mean_value{ 0.f, 0.f };
    if (i==0 && 0<j && j<Ny/2){
        // printf("%d, %d \n",i,j);
        int index2 = (Ny-j)*Nxh + i;
        mean_value = 0.5f * (f_spec[index] + cuConjf(f_spec[index2]));
        f_spec[index] = mean_value;
        f_spec[index2] = mean_value;
    }
	// for( int y{(index+1)*Nxh}; y<(Ny/2*Nxh); y+=stride*Nxh )
	// {
	// 	mean_value = 0.5f * ( w_new_comp[y] + cuConjf(w_new_comp[size-y]) );
	// 	w_new_comp[y] = mean_value;
	// 	w_new_comp[size-y] = cuConjf(mean_value);
	// }
}

//update the spectral space based on the value in physical space
void FwdTrans(Mesh* pmesh, float* phys, cuComplex* spec){
    cufft_error_func( cufftExecR2C(pmesh->transf, phys, spec));
    cuda_error_func( cudaDeviceSynchronize() );
}

//update the physics space based on the value in spectral space
void BwdTrans(Mesh* pmesh, cuComplex* spec, float* phys){
    int Nx = pmesh->Nx;
    int Nxh = pmesh->Nxh;
    int Ny = pmesh->Ny;
    int BSZ = pmesh->BSZ;
    // dim3 dimGrid = pmesh->dimGridsp;
    // dim3 dimBlock = pmesh->dimBlocksp;
    float* cutoff = pmesh->cutoff;
    // dealiasing_func<<<dimGrid, dimBlock>>>(spec, cutoff, Nxh, Ny, BSZ);
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

// pc = a*pa + b
__global__
void FldAdd(float a, float* pa, float b, float* pc, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b;
    }
}

// set physical field equals to a constant field
__global__
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

// spectral multiplication: spc(k,l) = C*spa(k,l)*spb(k,l)
__global__ 
void SpecMul(cuComplex* spa, cuComplex* spb, float C, cuComplex*spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = C*spa[index]*spb[index];
    }
}
__global__ 
void SpecMul(cuComplex* spa, cuComplex* spb, cuComplex C, cuComplex*spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = C*spa[index]*spb[index];
    }
}
__global__ 
void SpecMul(float* spa, cuComplex* spb, float C, cuComplex*spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = C*spa[index]*spb[index];
    }
}
__global__ 
void SpecMul(cuComplex* spa, float* spb, float C, cuComplex*spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = C*spa[index]*spb[index];
    }
}
__global__ 
void SpecMul(float* spa, cuComplex* spb, cuComplex C, cuComplex*spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = C*spa[index]*spb[index];
    }
}
__global__ 
void SpecMul(cuComplex* spa, float* spb, cuComplex C, cuComplex*spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = C*spa[index]*spb[index];
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

__global__ 
void laplacian_funcD(cuComplex *ft, cuComplex *lft, float* k_squared, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nxh;
    if(i<Nxh && j<Ny){
        // \hat{Dxx(u) + Dyy(u)} = -1*(kx^2 + ky^2)*\hat{u}
        lft[index] = -1* k_squared[index]*ft[index]; 
    }
}
void laplacian_func(cuComplex *ft, cuComplex *lft, Mesh* mesh){
    laplacian_funcD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(ft, lft, mesh->k_squared, 
    mesh->Nxh, mesh->Ny, mesh->BSZ);
}


__global__ 
void reality_func(cuComplex *spec, int Nxh, int Ny, int BSZ);

