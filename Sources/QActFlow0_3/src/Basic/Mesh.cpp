#include <Basic/Mesh.h>

Mesh::Mesh(int pBSZ, int pNx, int pNy, Qreal pLx, Qreal pLy):BSZ(pBSZ),
Nx(pNx), Ny(pNy), Lx(pLx), Ly(pLy), Nxh(pNx/2+1),dx(2*M_PI/pNx), 
dy(2*M_PI/pNy),alphax(2*M_PI/pLx),alphay(2*M_PI/pLy){
    cufft_error_func( cufftPlan2d( &(this->transf), Ny, Nx, CUFFT_D2Z ) );
    cufft_error_func( cufftPlan2d( &(this->inv_transf), Ny, Nx, CUFFT_Z2D ) );

    cuda_error_func(cudaMallocManaged( &(this->kx), sizeof(Qreal)*(Nx)));
    cuda_error_func(cudaMallocManaged( &(this->ky), sizeof(Qreal)*(Ny)));
    cuda_error_func(cudaMallocManaged( &(this->k_squared), sizeof(Qreal)*(Ny*Nxh)));
    cuda_error_func(cudaMallocManaged( &(this->cutoff), sizeof(Qreal)*(Ny*Nxh)));

    for (int i=0; i<Nxh; i++)          
    {
        this->kx[i] = i*alphax;
    } 
    for (int j=0; j<Ny; j++)          
    {
        if(j<+Nx/2+1)
        this->ky[j] = j*alphay;
        else 
        this->ky[j] = (j-Ny)*alphay;
    } 
    // the k^2 = kx^2+ky^2
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int c = i + j*Nxh;
            this->k_squared[c] = kx[i]*kx[i] + ky[j]*ky[j];
        }
    }

    // set up the cutoff
    int lowNy = Ny/3;
    int lowNx = Nx/3;
    for (int j=0; j<Ny; j++){
        for (int i=0; i<Nxh; i++){
            int c = i + j*Nxh;
            // set to be 1
            // -lowNy < ky <= lowNy && -lowNx < kx <= lowNx
            // (ky <= lowNy or Ny - ky > -lowNy) && (kx <= lowNx)
            if ( ( (j<= lowNy) || (Ny - j)< lowNy ) && (i<=lowNx) ){
                cutoff[c] = 1.f;
            }
            else{
                cutoff[c] = 0.f;
            }
        }
    }

    // thread information for physical space
    dimGridp = dim3(int((Nx-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dimBlockp = dim3(BSZ, BSZ);

    // thread information for spectral space
    dimGridsp = dim3(int((Nxh-0.5)/BSZ) + 1, int((Ny-0.5)/BSZ) + 1);
    dimBlocksp = dim3(BSZ, BSZ);
}
Mesh::~Mesh(){
    cuda_error_func(cudaFree(this->kx));
    cuda_error_func(cudaFree(this->ky));
    cuda_error_func(cudaFree(this->k_squared));
    cuda_error_func(cudaFree(this->cutoff));
};