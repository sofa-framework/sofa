#ifndef SOFA_CORE_MASS_H
#define SOFA_CORE_MASS_H

#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{

class Mass : public virtual Abstract::BaseObject
{
public:
    virtual ~Mass() { }

    virtual void addMDx()=0; ///< f += M dx

    virtual void accFromF()=0; ///< dx = M^-1 f

    virtual void computeForce() { }

    virtual void computeDf() { }
};

} // namespace Core

} // namespace Sofa

#endif
