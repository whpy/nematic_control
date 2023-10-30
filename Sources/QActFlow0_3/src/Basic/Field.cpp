#include <Basic/Field.h>

Field::Field(Mesh* pMesh): mesh(pMesh){
    cuda_error_func(cudaMallocManaged(&(this->phys), sizeof(Qreal)*((mesh->Nx)*(mesh->Ny))));
    cuda_error_func(cudaMallocManaged(&(this->spec), sizeof(Qcomp)*((mesh->Ny)*(mesh->Nxh))));
}

Field::~Field(){
    cudaFree(this->spec);
    cudaFree(this->phys);
}
