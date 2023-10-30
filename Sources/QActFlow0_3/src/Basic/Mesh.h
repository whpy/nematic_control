#ifndef __MESH_H
#define __MESH_H

#include <Basic/QActFlowDef.hpp>
#include <Basic/cuComplexBinOp.hpp>
#include <Basic/QActFlow.h>

class Mesh{ 
public:
    // the parameters about the block size
    int BSZ;
    // the basic parameters characterize the domain
    int Nx; int Ny; int Nxh;
    Qreal Lx; Qreal Ly;

    Qreal dx; Qreal dy;
    Qreal *kx; Qreal *ky; Qreal* k_squared;
    // \alpha_{i} = \frac{2\pi}{L_{i}}, which determines the wavenumber factor while deriving
    Qreal alphax; Qreal alphay; 
    // filter to cut off the high frequencies
    Qreal* cutoff;

    cufftHandle transf;
    cufftHandle inv_transf;

    // thread information for physical space 
    dim3 dimGridp;
    dim3 dimBlockp;
    // thread information for spectral space
    dim3 dimGridsp;
    dim3 dimBlocksp;
    
    Mesh(int BSZ, int pNx, int pNy, Qreal pLx, Qreal pLy);
    ~Mesh(void);
};

#endif // end of Mesh.h