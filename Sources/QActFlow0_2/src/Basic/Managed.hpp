# ifndef __MANAGED_HPP
#  define __MANAGED_HPP

# include <Basic/QActFlow.h>
class Managed{
    void *operator new(size_t len){
        void *p;
        cudaMallocManaged(&p, len);
        return p;
    }
};

# endif