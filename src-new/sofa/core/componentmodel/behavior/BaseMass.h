#ifndef SOFA_CORE_BASICMASS_H
#define SOFA_CORE_BASICMASS_H

#include "Sofa-old/Abstract/BaseObject.h"

namespace Sofa
{

namespace Core
{

class BasicMass : public virtual Abstract::BaseObject
{
public:
    virtual ~BasicMass() { }

    virtual void addMDx() = 0; ///< f += M dx using dof->getF() and dof->getDx()

    virtual void accFromF() = 0; ///< dx = M^-1 f using dof->getF() and dof->getDx()

    virtual double getKineticEnergy() = 0;  ///< vMv/2 using dof->getV()

};

} // namespace Core

} // namespace Sofa

#endif
