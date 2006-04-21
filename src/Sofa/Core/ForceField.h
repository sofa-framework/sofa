#ifndef SOFA_CORE_FORCEFIELD_H
#define SOFA_CORE_FORCEFIELD_H

#include <iostream>

#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{

class ForceField : public virtual Abstract::BaseObject
{
public:
    virtual ~ForceField() {}

    virtual void addForce () = 0;

    virtual void addDForce () = 0;
};

} // namespace Core

} // namespace Sofa

#endif
