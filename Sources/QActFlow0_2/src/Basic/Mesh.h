#ifndef __MESH_H
#define __MESH_H

#include <Basic/QActFlow.h>
#include <Basic/Managed.hpp>

#define M_PI 3.1415926535897932384
class Mesh{ 
public:
    // the parameters about the block size
    int BSZ;
    // the basic parameters characterize the domain
    int Nx; int Ny; int Nxh;
    float Lx; float Ly;

    float dx; float dy;
    float *kx; float *ky; float* k_squared;
    // \alpha_{i} = \frac{2\pi}{L_{i}}, which determines the wavenumber factor while deriving
    float alphax; float alphay; 

    cufftHandle transf;
    cufftHandle inv_transf;

    // thread information for physical space 
    dim3 dimGridp;
    dim3 dimBlockp;
    // thread information for spectral space
    dim3 dimGridsp;
    dim3 dimBlocksp;
    
    Mesh(int BSZ, int pNx, int pNy, float pLx, float pLy);
    ~Mesh(void);
};

#endif // end of Mesh.h