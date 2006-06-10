#ifndef SOFA_CORE_INTERACTIONFORCEFIELD_H
#define SOFA_CORE_INTERACTIONFORCEFIELD_H

#include "ForceField.h"

namespace Sofa
{

namespace Core
{

class InteractionForceField : public BasicForceField
{
public:
    virtual BasicMechanicalModel* getMechModel1() = 0;
    virtual BasicMechanicalModel* getMechModel2() = 0;
};

} // namespace Core

} // namespace Sofa

#endif
