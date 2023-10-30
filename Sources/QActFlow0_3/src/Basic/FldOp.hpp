#pragma once 
#ifndef __FLDOP_HPP
#define __FLDOP_HPP

#include <Basic/FldOp.h>

////////////////////////////////////////////////////////////////////////////
//Basic Functions of transformations between physical and spectral spaces//
//////////////////////////////////////////////////////////////////////////
__global__ 
inline void coeff(Qreal *f, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        f[index] = f[index]/(Nx*Ny);
    }
}

__global__
inline void symmetry_func(Qcomp f_spec[], int Nxh, int Ny, int BSZ)
{
	int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
	
	// We make sure that the transformed array keeps the necessary symmetry for
	// RFT. (The 1st column should consist of pairs of complex conjugates.)
	// the first column due to the zero value of wavenumver in x, the sequence
	// degenerates to the trivial one-dimensional dft which need to fulfil symmetry
	// here size equal to (Nx/2+1)*Ny where the size of wave numbers.
	Qcomp mean_value{ 0.f, 0.f };
    if (i==0 && 0<j && j<Ny/2){
        // printf("%d, %d \n",i,j);
        int index2 = (Ny-j)*Nxh + i;
        mean_value = 0.5f * (f_spec[index] + cuConj(f_spec[index2]));
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
inline void FwdTrans(Mesh* pmesh, Qreal* phys, Qcomp* spec){
    cufft_error_func( cufftExecD2Z(pmesh->transf, phys, spec));
}

//update the physics space based on the value in spectral space
inline void BwdTrans(Mesh* pmesh, Qcomp* spec, Qreal* phys){
    int Nx = pmesh->Nx;
    int Nxh = pmesh->Nxh;
    int Ny = pmesh->Ny;
    int BSZ = pmesh->BSZ;
    Qreal* cutoff = pmesh->cutoff;
    // adjust the spectral to satisfy necessary constraints(symmetry, dealiasing only 
    // acts after the nonlinear operations)
    symmetry_func<<<pmesh->dimGridsp, pmesh->dimBlocksp>>>(spec,Nxh, Ny, BSZ);

    cufft_error_func( cufftExecZ2D(pmesh->inv_transf, spec, phys));
    coeff<<<pmesh->dimGridp, pmesh->dimBlockp>>>(phys, Nx, Ny, BSZ);
    // in the referenced source code, they seem a little bit abuse synchronization, this
    // may be a point that we could boost the performance in the future. we temporarily
    // comply with the same principle that our code would at least perform no worse than theirs
}

// __Device(spec): deprecate the high frequencies determined by the cutoff array
__global__
void dealiasing_func(Qcomp* f_spec, Qreal* cutoff,int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if (i<Nxh && j<Ny){
        f_spec[index] = cutoff[index]*f_spec[index];
    }
}
////////////////////////////END///////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////
//FUNCTIONS TO assist field and spectrum operating//
///////////////////////////////////////////////////////////////////////////

// __Device: phys Field multiplication: pc(x) = C*pa(x,y)*pb(x,y). prefix p denotes physical
__global__ 
inline void FldMul(Qreal* pa, Qreal* pb, Qreal C, Qreal* pc, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = C*pa[index]*pb[index];
    }
}

// __Device: phys Field addition: pc(x,y) = a*pa(x,y) + b*pb(x,y). prefix p denotes physical
__global__  
inline void FldAdd(Qreal a, Qreal* pa, Qreal b, Qreal* pb, Qreal* pc, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b*pb[index];
    }
}

// __Device: pc = a*pa + b
__global__
inline void FldAdd(Qreal a, Qreal* pa, Qreal b, Qreal* pc, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pc[index] = a*pa[index] + b;
    }
}

// __Device: set physical Field equals to a constant Field
__global__
inline void FldSet(Qreal * pf, Qreal c, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pf[index] = c;
    }
}

// __Device: set two physical Field equals
__global__
inline void FldSet(Qreal * pf, Qreal* c, int Nx, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nx + i;
    if(i<Nx && j<Ny){
        pf[index] = c[index];
    }
}

// __Device: set two spectrum equal
__global__
inline void SpecSet(Qcomp * pa, Qcomp* pb, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        pa[index] = pb[index];
    }
}

// __Device: set a constant spectrum 
__global__
inline void SpecSet(Qcomp * pa, Qcomp c, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        pa[index] = c;
    }
}


// spectral addition: spc(k,l) = a*spa(k,l) + b*spb(k,l)
__global__
inline void SpecAdd(Qreal a, Qcomp* spa, Qreal b, Qcomp* spb, Qcomp* spc, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        spc[index] = a*spa[index] + b*spb[index];
    }
}



////////////////////////////////////////////////////////////////////////////
//FUNCTIONS ABOUT DERIVATIVES//
///////////////////////////////////////////////////////////////////////////
__global__ void xDerivD(Qcomp *ft, Qcomp *dft, Qreal* kx, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*kx[i];
    }
}
void xDeriv(Qcomp *ft, Qcomp *dft, Mesh *mesh){
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    xDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->kx, mesh->Nxh, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__ void yDerivD(Qcomp *ft, Qcomp *dft, Qreal* ky, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = j*Nxh + i;
    if(i<Nxh && j<Ny){
        dft[index] = ft[index]*im()*ky[j];
    }
}
void yDeriv(Qcomp *ft, Qcomp *dft, Mesh *mesh){
    dim3 dimGrid = mesh->dimGridp;
    dim3 dimBlock = mesh->dimBlockp;
    yDerivD<<<dimGrid, dimBlock>>>(ft,dft,mesh->ky,mesh->Nxh, mesh->Ny, mesh->BSZ);
    cuda_error_func( cudaPeekAtLastError() );
	cuda_error_func( cudaDeviceSynchronize() );
}

__global__ 
void laplacian_funcD(Qcomp *ft, Qcomp *lft, Qreal* k_squared, int Nxh, int Ny, int BSZ){
    int i = blockIdx.x * BSZ + threadIdx.x;
    int j = blockIdx.y * BSZ + threadIdx.y;
    int index = i + j*Nxh;
    if(i<Nxh && j<Ny){
        // \hat{Dxx(u) + Dyy(u)} = -1*(kx^2 + ky^2)*\hat{u}
        lft[index] = -1.f* k_squared[index]*ft[index]; 
    }
}
void laplacian_func(Qcomp *ft, Qcomp *lft, Mesh* mesh){
    laplacian_funcD<<<mesh->dimGridsp, mesh->dimBlocksp>>>(ft, lft, mesh->k_squared, 
    mesh->Nxh, mesh->Ny, mesh->BSZ);
}


#endif