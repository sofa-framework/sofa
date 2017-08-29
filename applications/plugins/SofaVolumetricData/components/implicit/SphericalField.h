#ifndef IMPLICIT_SPHERE_H
#define IMPLICIT_SPHERE_H

#include "ScalarField.h"

namespace sofa
{

namespace component
{

namespace implicit
{

class SphericalField : public ScalarField {

public:
    SOFA_CLASS(SphericalField, BaseObject);
    SphericalField() { }
    virtual ~SphericalField() { }
    virtual double eval(Vector3 p) override ;
};

} /// implicit

using implicit::SphericalField ;

} /// component

} /// sofa

#endif
