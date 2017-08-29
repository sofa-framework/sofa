#ifndef IMPLICIT_SHAPE_H
#define IMPLICIT_SHAPE_H

#include <sofa/core/objectmodel/BaseObject.h>

namespace sofa
{

namespace component
{

namespace implicit
{

using sofa::core::objectmodel::BaseObject ;
using sofa::defaulttype::Vector3 ;

class ScalarField : public BaseObject {

public:
    ScalarField() { }
    virtual ~ScalarField() { }
    virtual double eval(Vector3 p) = 0;
};

}

using implicit::ScalarField ;

} /// component

} /// sofa

#endif // IMPLICIT_SHAPE
