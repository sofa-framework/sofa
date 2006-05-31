#ifndef SOFA_CORE_BASICFORCEFIELD_H
#define SOFA_CORE_BASICFORCEFIELD_H

#include "Sofa/Abstract/BaseObject.h"

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
};

} // namespace Core

} // namespace Sofa

#endif
