#ifndef __FIELD_H
#define __FIELD_H

#include <Basic/QActFlow.h>
#include <Basic/Managed.hpp>
#include <Basic/Mesh.h>

class Field {
public:
    Mesh* mesh;
    float* phys;
    cuComplex* spec;
    Field(Mesh* pMesh);
    ~Field();
};

#endif