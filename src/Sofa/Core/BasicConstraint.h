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

    virtual void applyConstraint() = 0; ///< project dx to constrained space

    virtual BasicMechanicalModel* getDOFs() { return NULL; }
};

} // namespace Core

} // namespace Sofa

#endif
