#ifndef SOFA_CORE_BASICMASS_H
#define SOFA_CORE_BASICMASS_H

#include "Sofa/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{

class BasicMass : public virtual Abstract::BaseObject
{
public:
    virtual ~BasicMass() { }

    virtual void addMDx() = 0; ///< f += M dx

    virtual void accFromF() = 0; ///< dx = M^-1 f

    // Note: computeForce and computeDf are now replaced by addForce and addDForce of the ForceField class.

    //virtual void computeForce() { } ///< f += gravity and inertia forces

    //virtual void computeDf() { }
};

} // namespace Core

} // namespace Sofa

#endif
