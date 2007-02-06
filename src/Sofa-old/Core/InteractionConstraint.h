#ifndef SOFA_CORE_INTERACTIONCONSTRAINT_H
#define SOFA_CORE_INTERACTIONCONSTRAINT_H

#include "BasicConstraint.h"

namespace Sofa
{

namespace Core
{

class InteractionConstraint : public BasicConstraint
{
public:
    virtual BasicMechanicalModel* getMechModel1() = 0;
    virtual BasicMechanicalModel* getMechModel2() = 0;
};

} // namespace Core

} // namespace Sofa

#endif
