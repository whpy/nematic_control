#ifndef __FIELD_H
#define __FIELD_H

#include <Basic/QActFlow.h>
#include <Basic/Mesh.h>

class Field {
public:
    Mesh* mesh;
    Qreal* phys;
    Qcomp* spec;
    Field(Mesh* pMesh);
    ~Field();
};

#endif