#ifndef SOFA_CORE_BASICFORCEFIELD_H
#define SOFA_CORE_BASICFORCEFIELD_H

#include "Sofa/Abstract/BaseObject.h"
#include "Sofa/Components/Common/SofaBaseMatrix.h"
#include "Sofa/Components/Common/SofaBaseVector.h"

namespace Sofa
{

namespace Core
{

class BasicForceField : public virtual Abstract::BaseObject
{
public:
    virtual ~BasicForceField() {}

    virtual void addForce() = 0;

    virtual void addDForce() = 0;

    virtual void computeMatrix(Sofa::Components::Common::SofaBaseMatrix *, double , double , double, unsigned int &) {};

    virtual void contributeToMatrixDimension(unsigned int * const, unsigned int * const) {};

    virtual void computeVector(Sofa::Components::Common::SofaBaseVector *, unsigned int &) {};

    virtual void matResUpdatePosition(Sofa::Components::Common::SofaBaseVector *, unsigned int &) {};
};

} // namespace Core

} // namespace Sofa

#endif
