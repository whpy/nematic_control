#include <Basic/QActFlow.h>
#include <Field/Field.h>

Field::Field(Mesh* pMesh): mesh(pMesh){
    cuda_error_func(cudaMallocManaged(&(this->phys), sizeof(float)*((mesh->Nx)*(mesh->Ny))));
    cuda_error_func(cudaMallocManaged(&(this->spec), sizeof(cuComplex)*((mesh->Ny)*(mesh->Nxh))));
}

Field::~Field(){
    cudaFree(this->spec);
    cudaFree(this->phys);
}

