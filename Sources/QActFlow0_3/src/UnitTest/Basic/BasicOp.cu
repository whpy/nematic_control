#include <Basic/cuComplexBinOp.hpp>
#include <Basic/QActFlowDef.hpp>
#include <stdio.h>

using namespace std;
int main(){
    Qreal x = 1.0;
    Qcomp y = {1.,1.};
    Qcomp z = x + y;
    printf("x = %lf \n",x);
    printf("y = (%lf,%lf) \n",y.x, y.y);
    printf("x + y = z = (%lf,%lf) \n",z.x, z.y);
    printf("\n");

    x = 5.0;
    y = {-1.,1.};
    z = x - y;
    printf("x = %lf \n",x);
    printf("y = (%lf,%lf) \n",y.x, y.y);
    printf("x - y = z = (%lf,%lf) \n",z.x, z.y);
    printf("\n");

    x = 3.5;
    y = {2.,2.};
    z = x * y;
    printf("x = %lf \n",x);
    printf("y = (%lf,%lf) \n",y.x, y.y);
    printf("x * y = z = (%lf,%lf) \n",z.x, z.y);
    printf("\n");

    x = 8.0;
    y = {2.,2.};
    z = x / y;
    printf("x = %lf \n",x);
    printf("y = (%lf,%lf) \n",y.x, y.y);
    printf("x / y = z = (%lf,%lf) \n",z.x, z.y);
    printf("\n");

    Qcomp x1 = {-1., 1.};
    Qcomp y1 = {1.,1.};
    Qcomp z1 = x1 + y1;
    printf("x = (%lf,%lf) \n",x1.x, x1.y);
    printf("y = (%lf,%lf) \n",y1.x, y1.y);
    printf("x + y = z = (%lf,%lf) \n",z1.x, z1.y);
    printf("\n");

    x1 = {5., 6.};
    y1 = {1.,1.};
    z1 = x1 - y1;
    printf("x = (%lf,%lf) \n",x1.x, x1.y);
    printf("y = (%lf,%lf) \n",y1.x, y1.y);
    printf("x - y = z = (%lf,%lf) \n",z1.x, z1.y);
    printf("\n");

    x1 = {3.2, 6.4};
    y1 = {1.3,1.6};
    z1 = x1 * y1;
    printf("x = (%lf,%lf) \n",x1.x, x1.y);
    printf("y = (%lf,%lf) \n",y1.x, y1.y);
    printf("x * y = z = (%lf,%lf) \n",z1.x, z1.y);
    printf("\n");

    x1 = {5.5, 6.2};
    y1 = {1.8,8.2};
    z1 = x1 / y1;
    printf("x = (%lf,%lf) \n",x1.x, x1.y);
    printf("y = (%lf,%lf) \n",y1.x, y1.y);
    printf("x / y = z = (%.10lf,%.10lf) \n",z1.x, z1.y);
    printf("\n");
    // Qcomp y = {1.,1.};
    // Qcomp z = x + y;
    // printf("x = %lf \n",x);
    // printf("y = (%lf,%lf) \n",y.x, y.y);
    // printf("x + y = z = (%lf,%lf) \n",z.x, z.y);Qreal x = 1.0;

    // Qcomp y = {1.,1.};
    // Qcomp z = x + y;
    // printf("x = %lf \n",x);
    // printf("y = (%lf,%lf) \n",y.x, y.y);
    // printf("x + y = z = (%lf,%lf) \n",z.x, z.y);
    return 0;
}