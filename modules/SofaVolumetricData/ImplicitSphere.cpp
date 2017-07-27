#include "ImplicitSphere.h"

namespace sofa {

namespace core {


//factory register
int ImplicitSphereComponent = sofa::core::RegisterObject("Use to store grid").add< ImplicitSphere >();


double ImplicitSphere::eval(Coord p) {

    double x=p.x(), y=p.y(), z=p.z();
    double x2=x*x, y2=y*y, z2=z*z;
    double x4=x2*x2, y4=y2*y2, z4=z2*z2;
    return x4  + y4  + z4  + 2 *x2*  y2  + 2* x2*z2  + 2*y2*  z2  - 5 *x2  + 4* y2  - 5*z2+4;

}

}

}
