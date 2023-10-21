#ifndef __IF_CUH
#define __IF_CUH

#include <Basic/Mesh.h>
#include <Basic/QActFlow.h>

class IF{
public:
    Mesh* mesh;
    cuComplex alpha;

    IF(Mesh* pmesh, cuComplex palpha):mesh(pmesh), alpha(palpha){
    }

};

#endif