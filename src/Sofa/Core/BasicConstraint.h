#ifndef SOFA_CORE_BASICCONSTRAINT_H
#define SOFA_CORE_BASICCONSTRAINT_H

#include "Sofa/Abstract/BaseObject.h"
#include "Sofa/Core/BasicMechanicalModel.h"

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

    virtual BasicMechanicalModel* getDOFs() { return NULL; }
};

} // namespace Core

} // namespace Sofa

#endif
