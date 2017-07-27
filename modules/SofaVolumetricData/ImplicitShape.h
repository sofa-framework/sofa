#ifndef IMPLICIT_SHAPE_H
#define IMPLICIT_SHAPE_H

#include <iostream>
#include <sofa/core/objectmodel/BaseObject.h>
#include <sofa/core/ObjectFactory.h>

namespace sofa
{

namespace core
{

using namespace sofa::core::objectmodel;
typedef sofa::defaulttype::Vector3 Coord;

class ImplicitShape : public BaseObject {

public:
    ImplicitShape() { }
    virtual ~ImplicitShape() { }
    virtual double eval(Coord p) = 0;

};

}

}

#endif
