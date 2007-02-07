#ifndef SOFA_CORE_BASICCONSTRAINT_H
#define SOFA_CORE_BASICCONSTRAINT_H

#include "Sofa-old/Abstract/BaseObject.h"
#include "Sofa-old/Core/BasicMechanicalModel.h"

#include "Sofa-old/Components/Common/SofaBaseMatrix.h"
#include "Sofa-old/Components/Common/SofaBaseVector.h"

#include <vector>

namespace Sofa
{

namespace Core
{

class BasicConstraint : public virtual Abstract::BaseObject
{
public:
    virtual ~BasicConstraint() { }

    virtual void projectResponse() = 0; ///< project dx to constrained space (dx models an acceleration)
    virtual void projectVelocity() = 0; ///< project dx to constrained space (dx models a velocity)
    virtual void projectPosition() = 0; ///< project x to constrained space (x models a position)

    virtual void applyConstraint() {};

    virtual void applyConstraint(Components::Common::SofaBaseMatrix *, unsigned int &) {};
    virtual void applyConstraint(Components::Common::SofaBaseVector *, unsigned int &) {};

    virtual BasicMechanicalModel* getDOFs() { return NULL; }

    virtual void getConstraintValue(double *) {};
};

} // namespace Core

} // namespace Sofa

#endif
