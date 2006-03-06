#ifndef SOFA_CORE_FORCEFIELD_H
#define SOFA_CORE_FORCEFIELD_H

#include <iostream>

#include "Sofa/Abstract/Base.h"

namespace Sofa
{

namespace Core
{

class ForceField : public virtual Abstract::Base
{
public:
    virtual ~ForceField() {}

    virtual void addForce () = 0;

    virtual void addDForce () = 0;

    virtual void init () { };
};

} // namespace Core

} // namespace Sofa

#endif
