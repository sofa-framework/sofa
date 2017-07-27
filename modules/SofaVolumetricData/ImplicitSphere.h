#ifndef IMPLICIT_SPHERE_H
#define IMPLICIT_SPHERE_H

#include <SofaVolumetricData/ImplicitShape.h>

namespace sofa
{

namespace core
{

using namespace sofa::core::objectmodel;
typedef sofa::defaulttype::Vector3 Coord;

class ImplicitSphere : public ImplicitShape {

public:
    SOFA_CLASS(ImplicitSphere, BaseObject);
    ImplicitSphere() { }
    virtual ~ImplicitSphere() { }
    virtual double eval(Coord p);

};

}

}

#endif
